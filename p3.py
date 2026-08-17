import os
import time
import json
import glob
from pathlib import Path
from collections import Counter

import numpy as np

SEED = 42
N_FRAMES = 16
IMBALANCE_FACTOR = 100
MIN_CLASS_CLIPS = 8
MAX_CLASSES = 50

FEATURE_STORE = "/content/drive/MyDrive/ucf101_lt/clip_features.parquet"
RESULTS_JSON = "/content/drive/MyDrive/ucf101_lt/results.json"

def find_videos(root):
    """[(path, class_name)] for any nested layout. Parent dir is the label."""
    out = []
    for ext in ("*.avi", "*.mp4", "*.mkv"):
        for p in glob.glob(os.path.join(root, "**", ext), recursive=True):
            out.append((p, Path(p).parent.name))
    return sorted(out)


def make_long_tailed(items, imbalance_factor=IMBALANCE_FACTOR, seed=SEED):
    rng = np.random.RandomState(seed)
    by_class = {}
    for path, cls in items:
        by_class.setdefault(cls, []).append(path)

    too_small = {c: len(v) for c, v in by_class.items() if len(v) < MIN_CLASS_CLIPS}
    if too_small:
        print(f"      dropping {len(too_small)} class(es) with < {MIN_CLASS_CLIPS} "
              f"clips: {too_small}")
        by_class = {c: v for c, v in by_class.items() if len(v) >= MIN_CLASS_CLIPS}
    if not by_class:
        raise SystemExit("no class has enough clips")

    classes = sorted(by_class, key=lambda c: -len(by_class[c]))

    if len(classes) > MAX_CLASSES:
        pick = np.linspace(0, len(classes) - 1, MAX_CLASSES).astype(int)
        classes = [classes[i] for i in sorted(set(pick))]
        by_class = {c: by_class[c] for c in classes}
        print(f"      capped to {len(classes)} classes")

    C = len(classes)
    n_max = len(by_class[classes[0]])

    kept, counts = [], {}
    for i, cls in enumerate(classes):
        frac = (1.0 / imbalance_factor) ** (i / max(C - 1, 1))
        n_keep = min(max(int(round(n_max * frac)), MIN_CLASS_CLIPS), len(by_class[cls]))
        pool = by_class[cls]
        for j in rng.choice(len(pool), size=n_keep, replace=False):
            kept.append((pool[j], cls))
        counts[cls] = n_keep

    realised = max(counts.values()) / max(min(counts.values()), 1)
    if realised < imbalance_factor * 0.9:
        print(f"      imbalance factor: requested {imbalance_factor}, "
              f"realised {realised:.1f} (tail clamped by floor/pool size)")
    return kept, counts, classes, realised


def split_train_test(items, test_frac=0.3, seed=SEED):
    rng = np.random.RandomState(seed)
    by_class = {}
    for path, cls in items:
        by_class.setdefault(cls, []).append(path)

    train, test = [], []
    for cls, paths in by_class.items():
        idx = rng.permutation(len(paths))
        n_test = max(int(round(len(paths) * test_frac)), 1)
        test += [(paths[j], cls) for j in idx[:n_test]]
        train += [(paths[j], cls) for j in idx[n_test:]]
    return train, test

def sample_frames(path, n=N_FRAMES):
    import cv2
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    out, last = [], None
    for j in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(j))
        ok, frame = cap.read()
        if ok:
            last = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if last is not None:
            out.append(last)
    cap.release()

    if not out:
        return None
    if len(out) < n:
        out += [out[-1]] * (n - len(out))
    return np.stack(out[:n])


def extract_features(items):
    import torch
    import open_clip
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai")
    model = model.to(device).eval()

    feats, labels, paths = [], [], []
    t0 = time.time()
    for k, (path, cls) in enumerate(items):
        arr = sample_frames(path)
        if arr is None:
            continue
        batch = torch.stack([preprocess(Image.fromarray(f)) for f in arr]).to(device)
        with torch.no_grad():
            f = model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy().astype(np.float32))
        labels.append(cls)
        paths.append(path)
        if (k + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {k+1}/{len(items)}  {el:.0f}s  eta {el/(k+1)*(len(items)-k-1):.0f}s",
                  flush=True)

    return np.stack(feats), labels, paths, model


def save_feature_store(feats, labels, paths, out=FEATURE_STORE):
    import polars as pl
    n, t, d = feats.shape
    pl.DataFrame({
        "path": paths,
        "label": labels,
        "n_frames": [t] * n,
        "dim": [d] * n,
        "feat": [feats[i].reshape(-1).tolist() for i in range(n)],
    }).write_parquet(out)
    print(f"saved {out}  ({n} clips, {t} frames, {d} dim)")


def load_feature_store(path=FEATURE_STORE):
    import polars as pl
    df = pl.read_parquet(path)
    t, d = int(df["n_frames"][0]), int(df["dim"][0])
    feats = np.array(df["feat"].to_list(), dtype=np.float32).reshape(-1, t, d)
    return feats, df["label"].to_list(), df["path"].to_list()

def _humanise(name):
    """ApplyEyeMakeup -> apply eye makeup"""
    out = []
    for ch in name:
        if ch.isupper() and out:
            out.append(" ")
        out.append(ch.lower())
    return "".join(out).replace("_", " ").strip()


def zero_shot(feats, labels, classes, model=None):
    import torch
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai")
        model = model.to(device).eval()
    tok = open_clip.get_tokenizer("ViT-B-32")

    with torch.no_grad():
        tf = model.encode_text(
            tok(["a video of a person " + _humanise(c) for c in classes]).to(device))
        tf = tf / tf.norm(dim=-1, keepdim=True)

    pooled = torch.tensor(feats.mean(axis=1)).to(device)
    pooled = pooled / pooled.norm(dim=-1, keepdim=True)
    pred = (pooled @ tf.T).argmax(dim=1).cpu().numpy()
    return pred, np.array([classes.index(l) for l in labels])


class TemporalTransformer:
    def __init__(self, dim=512, n_classes=10, n_layers=2, n_heads=8, seed=SEED):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.cls = nn.Parameter(torch.zeros(1, 1, dim))
                self.pos = nn.Parameter(torch.zeros(1, N_FRAMES + 1, dim))
                layer = nn.TransformerEncoderLayer(
                    d_model=dim, nhead=n_heads, dim_feedforward=dim * 2,
                    dropout=0.1, batch_first=True, norm_first=True)
                self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
                self.head = nn.Linear(dim, n_classes)
                nn.init.trunc_normal_(self.pos, std=0.02)
                nn.init.trunc_normal_(self.cls, std=0.02)

            def forward(self, x):
                b = x.shape[0]
                x = torch.cat([self.cls.expand(b, -1, -1), x], dim=1)
                x = x + self.pos[:, : x.shape[1]]
                return self.head(self.enc(x)[:, 0])

        self.net = Net().to(self.device)

    def fit(self, X, y, class_weights=None, loss_type="ce", epochs=30, lr=1e-3):
        torch = self.torch
        import torch.nn.functional as F

        Xt = torch.tensor(X).to(self.device)
        yt = torch.tensor(y).long().to(self.device)
        w = None if class_weights is None else torch.tensor(
            class_weights, dtype=torch.float32).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=0.01)

        self.net.train()
        for _ in range(epochs):
            perm = torch.randperm(len(Xt), device=self.device)
            for i in range(0, len(perm), 64):
                b = perm[i: i + 64]
                logits = self.net(Xt[b])
                loss = (_focal_loss(logits, yt[b], weight=w) if loss_type == "focal"
                        else F.cross_entropy(logits, yt[b], weight=w))
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    def predict(self, X):
        with self.torch.no_grad():
            return self.net(self.torch.tensor(X).to(self.device)).argmax(dim=1).cpu().numpy()


def _focal_loss(logits, target, gamma=2.0, weight=None):
    import torch.nn.functional as F
    logp = F.log_softmax(logits, dim=-1)
    logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    loss = -((1 - logpt.exp()) ** gamma) * logpt
    if weight is not None:
        loss = loss * weight[target]
    return loss.mean()


def effective_number_weights(counts, beta=0.999):
    n = np.asarray(counts, dtype=np.float64)
    eff = (1.0 - np.power(beta, n)) / (1.0 - beta)
    w = 1.0 / np.maximum(eff, 1e-8)
    return w / w.sum() * len(n)

def _spearman(a, b):
    def rank(x):
        x = np.asarray(x, dtype=float)
        r = np.empty(len(x), dtype=float)
        r[x.argsort()] = np.arange(len(x), dtype=float)
        for v in np.unique(x):
            m = (x == v)
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def bias_audit(pred, y, classes, train_counts):
    per_class, n_eval = {}, {}
    for i, c in enumerate(classes):
        m = (y == i)
        n_eval[c] = int(m.sum())
        per_class[c] = float((pred[m] == y[m]).mean()) if m.sum() else float("nan")

    scored = {c: v for c, v in per_class.items() if not np.isnan(v)}
    if len(scored) < len(per_class):
        print(f"      {len(per_class) - len(scored)} class(es) had no test clips "
              f"and are excluded")
    if not scored:
        raise SystemExit("no class had test clips")

    order = sorted(scored, key=lambda c: -train_counts.get(c, 0))
    third = max(len(order) // 3, 1)
    head = [scored[c] for c in order[:third]]
    tail = [scored[c] for c in order[-third:]]
    accs = [scored[c] for c in order]
    freqs = [train_counts.get(c, 0) for c in order]

    return {
        "overall": float((pred == y).mean()),
        "macro": float(np.mean(accs)),
        "worst_group": float(np.min(accs)),
        "worst_class": min(scored, key=lambda c: scored[c]),
        "n_classes_scored": len(scored),
        "head_acc": float(np.mean(head)),
        "tail_acc": float(np.mean(tail)),
        "head_minus_tail": float(np.mean(head) - np.mean(tail)),
        "freq_acc_spearman": _spearman(freqs, accs) if len(accs) > 2 else float("nan"),
        "per_class": per_class,
        "n_test_per_class": n_eval,
    }


def mean_pool_baselines(feats, labels, is_train, classes, seeds=(0, 1, 2, 3, 4)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    Xtr, Xte = feats[is_train].mean(1), feats[~is_train].mean(1)
    ytr = np.array([classes.index(l) for l, t in zip(labels, is_train) if t])
    yte = np.array([classes.index(l) for l, t in zip(labels, is_train) if not t])
    train_counts = Counter([l for l, t in zip(labels, is_train) if t])

    def score(pred):
        a = bias_audit(pred, yte, classes, train_counts)
        return (a["overall"], a["macro"], a["head_acc"], a["tail_acc"],
                a["worst_group"], sum(v == 0 for v in a["per_class"].values()
                                      if not np.isnan(v)))

    out = {}
    out["logistic"] = score(
        LogisticRegression(max_iter=3000).fit(Xtr, ytr).predict(Xte))
    out["logistic_balanced"] = score(
        LogisticRegression(max_iter=5000, class_weight="balanced").fit(Xtr, ytr).predict(Xte))

    runs = [score(MLPClassifier(hidden_layer_sizes=(512,), max_iter=600,
                                random_state=s).fit(Xtr, ytr).predict(Xte))
            for s in seeds]
    r = np.array(runs)
    out["mlp"] = tuple(r.mean(axis=0))
    out["mlp_std"] = tuple(r.std(axis=0))
    out["mlp_runs"] = runs

    hdr = f"{'model':22s} {'overall':>8s} {'macro':>8s} {'head':>8s} {'tail':>8s} {'zeros':>6s}"
    print(hdr)
    for k in ["logistic", "logistic_balanced", "mlp"]:
        v = out[k]
        print(f"{k:22s} {v[0]:8.4f} {v[1]:8.4f} {v[2]:8.4f} {v[3]:8.4f} {v[5]:6.0f}")
    s = out["mlp_std"]
    print(f"{'mlp (sd over seeds)':22s} {s[0]:8.4f} {s[1]:8.4f} {s[2]:8.4f} {s[3]:8.4f}")
    return out


def benchmark_store(path=FEATURE_STORE, repeats=3):
    import polars as pl
    import pandas as pd

    def t(fn):
        best = float("inf")
        for _ in range(repeats):
            s = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - s)
        return best

    try:
        pl.scan_parquet(path).group_by("label").agg(pl.len()).collect()
        agg = pl.len()
    except Exception:
        agg = pl.count()

    t_pl = t(lambda: pl.scan_parquet(path).group_by("label").agg(agg.alias("n")).collect())
    t_naive = t(lambda: pd.read_parquet(path).groupby("label").size())
    t_fair = t(lambda: pd.read_parquet(path, columns=["label"]).groupby("label").size())

    return {"polars_s": t_pl, "pandas_naive_s": t_naive, "pandas_fair_s": t_fair,
            "speedup_vs_naive": t_naive / max(t_pl, 1e-9),
            "speedup_vs_fair": t_fair / max(t_pl, 1e-9)}

def run(video_root="/content/ucf101", max_per_class=None, out_tag="", epochs=30):
    print("[1/8] finding videos")
    items = find_videos(video_root)
    print(f"      {len(items)} clips, {len(set(c for _, c in items))} classes")
    if not items:
        raise SystemExit("no videos found")

    if max_per_class:
        seen, trimmed = {}, []
        for p, c in items:
            if seen.get(c, 0) < max_per_class:
                trimmed.append((p, c))
                seen[c] = seen.get(c, 0) + 1
        items = trimmed
        print(f"      trimmed to {len(items)} clips (max_per_class={max_per_class})")

    feature_store = FEATURE_STORE.replace(".parquet", f"{out_tag}.parquet")
    results_json = RESULTS_JSON.replace(".json", f"{out_tag}.json")

    print("[2/8] building long-tailed split")
    lt, counts, classes, realised_if = make_long_tailed(items)
    print(f"      {len(lt)} clips kept; largest {max(counts.values())}, "
          f"smallest {min(counts.values())}, imbalance factor {realised_if:.1f}")
    print("      distribution: " + ", ".join(f"{c}={counts[c]}" for c in classes))

    train, test = split_train_test(lt)

    print("[3/8] extracting CLIP features (once)")
    all_items = train + test
    feats, labels, paths, clip_model = extract_features(all_items)
    save_feature_store(feats, labels, paths, out=feature_store)

    train_paths = set(p for p, _ in train)
    is_tr = np.array([p in train_paths for p in paths])
    n_dropped = len(all_items) - len(paths)
    if n_dropped:
        print(f"      dropped {n_dropped} unreadable clips")

    Xtr, Xte = feats[is_tr], feats[~is_tr]
    ytr_lbl = [l for l, t in zip(labels, is_tr) if t]
    yte_lbl = [l for l, t in zip(labels, is_tr) if not t]
    ytr = np.array([classes.index(l) for l in ytr_lbl])
    yte = np.array([classes.index(l) for l in yte_lbl])
    train_counts = Counter(ytr_lbl)

    results = {"n_classes": len(classes), "n_train": len(ytr), "n_test": len(yte),
               "imbalance_factor_requested": IMBALANCE_FACTOR,
               "imbalance_factor_realised": realised_if,
               "n_dropped_unreadable": n_dropped,
               "train_counts": dict(train_counts)}

    print("[4/8] zero-shot CLIP baseline (no training)")
    zs_pred, zs_y = zero_shot(Xte, yte_lbl, classes, model=clip_model)
    results["zero_shot"] = bias_audit(zs_pred, zs_y, classes, train_counts)
    print(f"      overall {results['zero_shot']['overall']:.4f}  "
          f"macro {results['zero_shot']['macro']:.4f}")

    print("[5/8] temporal transformer, plain cross-entropy")
    m = TemporalTransformer(n_classes=len(classes)).fit(Xtr, ytr, epochs=epochs)
    results["ce"] = bias_audit(m.predict(Xte), yte, classes, train_counts)
    print(f"      overall {results['ce']['overall']:.4f}  "
          f"macro {results['ce']['macro']:.4f}  "
          f"worst {results['ce']['worst_group']:.4f}")

    print("[6/8] mitigations")
    w = effective_number_weights([train_counts.get(c, 0) for c in classes])
    m_cb = TemporalTransformer(n_classes=len(classes)).fit(
        Xtr, ytr, class_weights=w, epochs=epochs)
    results["class_balanced"] = bias_audit(m_cb.predict(Xte), yte, classes, train_counts)
    print(f"      class-balanced: macro {results['class_balanced']['macro']:.4f}  "
          f"worst {results['class_balanced']['worst_group']:.4f}")

    m_fl = TemporalTransformer(n_classes=len(classes)).fit(
        Xtr, ytr, loss_type="focal", epochs=epochs)
    results["focal"] = bias_audit(m_fl.predict(Xte), yte, classes, train_counts)
    print(f"      focal:          macro {results['focal']['macro']:.4f}  "
          f"worst {results['focal']['worst_group']:.4f}")

    print("[7/8] mean-pooled baselines (frame order discarded)")
    b = mean_pool_baselines(feats, labels, is_tr, classes)
    results["baselines"] = {k: list(v) if isinstance(v, tuple) else v
                            for k, v in b.items() if k != "mlp_runs"}

    print("[8/8] polars vs pandas on the feature store")
    b = benchmark_store(path=feature_store)
    results["benchmark"] = b
    print(f"      polars       {b['polars_s']*1000:8.1f} ms")
    print(f"      pandas naive {b['pandas_naive_s']*1000:8.1f} ms "
          f"({b['speedup_vs_naive']:.1f}x)")
    print(f"      pandas fair  {b['pandas_fair_s']*1000:8.1f} ms "
          f"({b['speedup_vs_fair']:.1f}x)")

    os.makedirs(os.path.dirname(results_json), exist_ok=True)
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {results_json}")
    return results


def smoke_test(video_root="/content/ucf101"):
    global MAX_CLASSES, MIN_CLASS_CLIPS, IMBALANCE_FACTOR
    saved = (MAX_CLASSES, MIN_CLASS_CLIPS, IMBALANCE_FACTOR)
    MAX_CLASSES, MIN_CLASS_CLIPS, IMBALANCE_FACTOR = 4, 4, 3
    try:
        return run(video_root, max_per_class=10, out_tag="_smoke", epochs=3)
    finally:
        MAX_CLASSES, MIN_CLASS_CLIPS, IMBALANCE_FACTOR = saved
