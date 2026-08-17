# Long-tailed video understanding on UCF-101

Frozen CLIP ViT-B/32 frame features, evaluated on a deliberately long-tailed
version of UCF-101.

The question isn't what accuracy is achievable. It's what happens to rare
classes when you train on an imbalanced catalog, and whether the standard fixes
for imbalance actually work.

**A plain MLP on mean-pooled frames beats a 2-block temporal transformer on
every metric, including a 24-point margin on rare classes - with no class
weighting at all. The transformer's apparent "imbalance amplification" was
over-parameterisation, not a property of supervised learning. That is also why
class-balanced reweighting and focal loss both failed: they target the loss
function, and the problem was model capacity.**

Discarding frame order (mean-pooling) costs nothing measurable here.

```
p3.py            pipeline
Project_3.ipynb  run + results
```

## Setup

| | |
|---|---|
| Source | UCF-101, 13,320 clips, 101 classes |
| Used | 50 classes, 1,913 clips |
| Imbalance | exponential decay, realised factor **20.9** |
| Largest / smallest class | 167 / 8 clips |
| Split | stratified 70/30, seed 42 |
| Encoder | frozen CLIP ViT-B/32, 16 uniform frames, mean-pooled |
| Head | 2-block transformer encoder, learned positional embeddings, CLS pooling |

Stock UCF-101 is close to balanced, so an imbalance audit on it finds nothing.
`make_long_tailed` resamples it with exponential decay (the CIFAR-LT recipe from
Cui et al. 2019). Classes with fewer than 8 clips are dropped - below that a
class gets one test clip and its accuracy is a coin flip, which makes
worst-group accuracy meaningless.

The realised imbalance factor comes out below the requested one because the
floor clamps the tail, so the code reports what it built (20.9) rather than what
it was asked for (100).

## Results

All models sit on the same frozen CLIP features and the same 70/30 split.

| Model | Params | Overall | Macro | Head | Tail | Zero-acc classes |
|---|---|---|---|---|---|---|
| **Mean-pool + MLP (512), unweighted** | ~0.3M | **0.9703** | **0.8951** | **0.9930** | **0.8375** | **2** |
| Mean-pool + balanced logistic | ~26K | 0.9489 | 0.8710 | 0.9684 | 0.7812 | 2 |
| Temporal transformer, plain CE | ~6M | 0.9506 | 0.8109 | 0.9915 | 0.5938 | 5 |
| Temporal transformer + class-balanced | ~6M | 0.9312 | 0.7855 | 0.9628 | 0.5625 | 5 |
| Temporal transformer + focal (γ=2) | ~6M | 0.9330 | 0.7764 | 0.9767 | 0.5625 | 5 |
| Mean-pool + plain logistic | ~26K | 0.8554 | 0.4653 | 0.9873 | 0.0312 | 23 |
| Zero-shot CLIP (no training) | 0 | 0.8025 | 0.7039 | 0.7905 | 0.5938 | - |

MLP figures are means over 5 seeds: **macro 0.8951 ± 0.0046, tail 0.8375 ±
0.0125**, head identical (0.9930) in all five. Transformer figures are single
runs - but the 0.244 tail gap is ~19× the MLP's seed-to-seed standard
deviation, so seed variance cannot account for it.

Head/tail are the top and bottom third of classes by training frequency.

Bootstrap over test clips, B=300:

| | Macro | Head | Tail |
|---|---|---|---|
| Zero-shot | 0.7083 [0.6639, 0.7608] | 0.7898 [0.7590, 0.8254] | 0.5882 [0.4524, 0.7328] |
| Trained | 0.8194 [0.7774, 0.8667] | 0.9915 [0.9814, 1.0000] | 0.5929 [0.4589, 0.7500] |

`tail(trained) − tail(zero-shot) = +0.0047, 95% CI [−0.2007, +0.1944]`.

The point estimates for both models are identical (0.5938), so the **measured
tail difference is exactly zero**; the +0.0047 is an artifact of bootstrap
resampling, not a measured gain. See "Validation of the evaluation itself".

## What the numbers say

**Capacity, not the loss function, drives rare-class failure.** The MLP has
roughly 20× fewer parameters than the transformer and beats it by 24 points on
the tail - using no class weighting at all. The transformer has ~6M parameters
and 1,346 training examples across 50 classes: 27 per class on average, 6 for
the rare ones. It memorises the head and has nothing left over.

**That is why both mitigations failed.** Class-balanced reweighting and focal
loss adjust *which examples the gradient attends to*. They cannot help a model
that lacks the sample-to-parameter ratio to generalise in the first place. The
unweighted MLP outperforms the weighted transformer on every metric, which is
the cleanest possible demonstration that the loss function was never the
binding constraint.

**Weighting matters enormously - for the right model.** On mean-pooled
features, switching plain logistic regression to class-balanced weighting takes
the tail from **0.0312 to 0.7812**. So class weighting is not useless; it was
being applied to a model whose problem it could not address.

**Aggregate accuracy still hides subgroup failure**, though less dramatically
once the model is right: 0.9703 overall against 0.8951 macro is a 7.5-point
gap, versus 14 points for the transformer.

**Discarding frame order costs nothing measurable.** Mean-pooling throws away
the sequence entirely and still wins. Whatever temporal structure a 2-block
transformer could exploit here is outweighed by its generalisation cost. This
is a statement about *this* dataset and *these* two untuned models, not about
temporal modelling in general.

**Zero-shot CLIP remains the useful control.** It never saw the training
distribution, so it cannot carry a class-frequency bias. Its tail accuracy is
0.5938. The transformer, after supervised training, also reaches 0.5938 - no
gain. The MLP reaches 0.8375. So supervised training *does* improve rare
classes substantially, by roughly 24 points over zero-shot, provided the model
is appropriately sized.

### Dead classes: 5 under the transformer, 2 under the MLP

| Class | Train n | Zero-shot | Transformer | MLP |
|---|---|---|---|---|
| YoYo | 8 | 0.00 | 0.00 | recovered |
| PommelHorse | 6 | 0.00 | 0.00 | recovered |
| RopeClimbing | 6 | 0.00 | 0.00 | recovered |
| StillRings | 6 | 0.00 | 0.00 | - |
| JavelinThrow | 6 | 0.50 | 0.00 | - |

Only **2** classes remain at zero under the MLP, in all five seeds. Three of the
five apparent "representation failures" were model failures: frozen CLIP *can*
separate them, the transformer just couldn't learn the boundary from six
examples.

This retires an earlier claim in this project that four classes were
irrecoverable because zero-shot CLIP also scored 0.00 on them. Scoring zero
under zero-shot turns out to be weak evidence about what a properly-sized
supervised head can do - the prompt template is a much cruder classifier than a
trained probe.

## Validation of the evaluation itself

The metrics were audited separately from the models. Four issues, all of which
widen the uncertainty rather than narrow it.

**1. The tail metric's resolution is 3.1 points.** Tail classes sit at the
8-clip floor, so each contributes 2 test clips and its per-class accuracy can
only be 0, 0.5 or 1. Tail accuracy is the mean of 16 such values, so it moves in
steps of 1/32 = **0.031**. This matters for the transformer-vs-zero-shot
comparison, where both scored exactly 19/32 and the difference was below the
representable minimum. It does not affect the MLP-vs-transformer comparison:
0.8375 vs 0.5938 is 7.8 resolution steps apart.

**2. The head/tail boundary falls inside a 19-way tie.** 19 classes share
train_n = 6. The tail takes the bottom 16 by training count, so 3 tied classes
are excluded by sort order alone, with no criterion distinguishing them.
Simulation over plausible per-class accuracies puts the resulting swing at
~0.15 on absolute tail accuracy and ~0.11 on the between-model difference. So
absolute tail figures should not be quoted to 4 decimal places. (Same failure
mode as the `PERCENT_RANK` tie artifact found in the companion causal project.)

**3. Bootstrap coverage is roughly 91–93%, not 95%.** Simulated against a known
ground truth with this exact class/clip structure, the nominal 95% interval
covered the truth 366/400 and 372/400 on two independent simulation runs.
Resampling clips i.i.d. under-propagates uncertainty for a metric that averages
over classes holding 2 clips each, so bootstrap intervals here are somewhat too
narrow. Note this figure is itself simulated, not measured on the trained
models.

**4. 18 of 50 classes are ignored.** Head and tail are the top and bottom third
by frequency, so the middle third contributes to macro accuracy but to neither
head nor tail.

Effect on the conclusions: issues 1–3 enlarge the uncertainty on any single
tail figure. They do **not** threaten the MLP-vs-transformer result, which is a
0.244 gap against a measured seed-to-seed standard deviation of 0.0125 - 19×
larger. They do mean no individual tail number should be quoted to four
decimals.

## Feature store

Features cache to Parquet so model iterations don't re-run CLIP.

| Query | Time |
|---|---|
| Polars, `scan_parquet` + group_by | 1.2 ms |
| pandas, `read_parquet(columns=["label"])` | 2.2 ms |
| pandas, naive `read_parquet()` | 665.1 ms |

The 560× gap against naive pandas is column pruning, not engine speed - a pandas
caller who passes `columns=` recovers nearly all of it. 1.8× against
equally well-written pandas is the honest comparison.

## Limitations

- **Neither model was hyperparameter-tuned.** The transformer used its defaults
  (2 blocks, d=512, 30 epochs, AdamW 1e-3) and the MLP used sklearn's. The
  honest claim is that under matched, untuned conditions the simpler model wins
  - not that transformers are unsuited to this task. A tuned, smaller
  transformer might well close the gap, and testing that is the obvious next
  step.
- **Transformer results are single runs**; only the MLP was seed-replicated
  (5 seeds). The 0.244 tail gap is 19× the MLP's seed standard deviation, so
  seed variance cannot plausibly explain it, but the transformer's own variance
  is unmeasured.
- **The MLP-vs-transformer comparison changes two things at once** - pooling
  and architecture. It shows a mean-pooled MLP wins; it does not isolate how
  much of that is discarding frame order versus reducing capacity. A
  mean-pooled transformer, or a sequence MLP, would separate them.
- **The tail rests on 32 test clips** across 16 classes, 2 clips each, so the
  metric's resolution is 3.1 points and no tail figure should be quoted to four
  decimals.
- **The tail/head split is tie-broken arbitrarily** - 19 classes share the same
  training count and only 16 fit in the tail. Fixing this properly means either
  reporting all classes below a frequency threshold rather than a fixed count,
  or averaging over tie-break choices.
- **The imbalance is constructed**, not naturally occurring.
- 50 of 101 classes, capped for runtime.
- Frozen encoder throughout. The 2 remaining zero-accuracy classes are the only
  candidates for genuine representation failure, and even that is unproven -
  fine-tuning CLIP was never attempted.
- Single zero-shot prompt template, no prompt ensembling, so the zero-shot
  numbers are probably a slight underestimate.
- No shot-boundary detection or frame-sampling ablation; 16 uniform frames
  throughout.
- Only two loss-based mitigations tested. Resampling and LDAM are untried,
  though the capacity finding suggests loss-based fixes are the wrong family of
  remedy for this failure.

## What changed, and why

An earlier version of this write-up claimed that supervised training amplifies
class imbalance and that neither standard mitigation helped. Both claims came
from comparing the transformer against zero-shot CLIP without ever running a
simple supervised baseline.

Adding that baseline reversed the conclusion. Training does not amplify
imbalance; an over-parameterised model does. The mitigations did not fail
because class imbalance is hard to fix; they failed because they address the
loss function and the binding constraint was model capacity.

The original claims are recorded here rather than deleted, because the
correction is the more useful result.

## References

- Cui et al., *Class-Balanced Loss Based on Effective Number of Samples*, CVPR 2019
- Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017
- Kang et al., *Decoupling Representation and Classifier for Long-Tailed Recognition*, ICLR 2020
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021
- Soomro et al., *UCF101: A Dataset of 101 Human Action Classes*, 2012
