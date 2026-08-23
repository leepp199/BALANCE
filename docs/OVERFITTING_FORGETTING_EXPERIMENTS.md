# Required experiments: overfitting control and anti-forgetting

These experiments are part of the paper, not optional implementation notes. Every
row uses the same retrained base checkpoint, stream order and ten reset repeats.

## Overfitting control

The base/meta training is evaluated on a held-out query stream, never on the support
episode used for optimization. We report train episode accuracy, held-out base
accuracy, open-set AUROC, FPR95 and calibration gap. The following controlled rows
separate mechanisms:

| Row | Configuration | Expected interpretation |
|---|---|---|
| A | no meta-training | closed-set representation/prototype reference |
| B | original frozen-encoder meta-training | effect of open-set hinge + FUNIT |
| C | B + held-out AUROC checkpoint selection | whether the original selection overfits episodes |
| D | C + layer-4 low-LR fine-tuning + base anchor | whether representation adaptation improves boundary without base drift |
| E | C + stronger dropout/weight decay/early stopping | conventional overfitting control |

An improvement is accepted only when held-out AUROC/FPR95 and base accuracy improve
together. A higher training AUROC alone is explicitly rejected as evidence.

## Incremental anti-forgetting

For each session we report:

* new-class accuracy (`inc`),
* all-seen accuracy (`all`),
* known-stream retention (`acc known`),
* old-class retention after expansion,
* forgetting `F_t = A_old,before - A_old,after`,
* new-class gain over the no-protection row.

The controlled rows are:

| Row | Configuration | Mechanism tested |
|---|---|---|
| F0 | prototype overwrite, no memory | reference failure |
| F1 | TEEN calibration only | few-shot prototype bias correction |
| F2 | F1 + covariance/statistical replay, no old write-back | synthetic hard examples with frozen history |
| F3 | F2 + confidence anchor + bounded residual update | limits noisy replay drift |
| F4 | F3 + guarded old-prototype adaptation | separates old-class compactness from novel absorption |
| F5 | F3 + DFSB/BCD discovery | tests discovery quality independently of retention |

The principle is measurable: F2/F3 should reduce the slope of old-class retention
loss across sessions, while F1/F5 should mainly improve new-class accuracy. If a row
raises `inc` but increases forgetting or lowers `all`, it is reported as a trade-off,
not as an overall improvement.

## Paper figures

The final paper will include (i) train-vs-held-out AUROC/base-accuracy curves for
A--D, (ii) session curves for `inc`, `all` and old retention for F0--F5, and (iii) a
two-axis scatter of new-class gain versus forgetting. All curves are generated from
raw per-repeat records, with mean ± 95% confidence intervals.
