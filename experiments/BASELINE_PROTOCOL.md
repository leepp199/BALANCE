# Three-dataset open-world class-incremental baseline protocol

## Datasets

| Dataset | Domain | Base classes | Incremental sessions | Total classes |
|---|---|---:|---:|---:|
| LS-100 | LibriSpeech speaker/audio classification | 80 | 4 x 5 | 100 |
| NS-100 | NSynth instrument/audio classification | 80 | 4 x 5 | 100 |
| FSC-89 | FSD-MIX-CLIPS sound-event classification | 69 | 4 x 5 | 89 |

## Fixed protocol

- Dataset-specific base-class count above; four 5-way incremental sessions; five stream samples per class.
- Mixed stream in every incremental session: previously seen and novel classes.
- Shared LS-100 split, shared acoustic encoder checkpoint, and shared random seeds.
- Ten independent repeats; classifier state is reset before every repeat.
- Report per-session known accuracy, discovery ACC, AUROC, incremental-only accuracy,
  all-seen accuracy, average accuracy, final-session accuracy, and performance degradation.

## Baselines

| ID | Method | Source status | LS-100 adaptation |
|---|---|---|---|
| B0 | Frozen prototype/KMeans | project baseline | none |
| B1 | DFSB structure discovery | project baseline | none |
| B2 | TEEN | official code, commit `2ba5165` | replace image encoder with shared acoustic features |
| B3 | VB-CGCD | official code, commit `7fe9501` | feed shared acoustic features into GMM + MNGMM |
| B4 | OFCL | official code, commit `033726f` | adapt ITA/MOB/AKS to acoustic feature tokens |
| B5 | OPCR | paper reimplementation | orthogonal targets + EVT rejection + confidence refinement |
| B6 | YLOC | paper reimplementation | prototype-centered base loss + prototype augmentation |

These methods form the recent/SOTA comparison set in the paper. “SOTA baseline” denotes
their role as recent competitive references, not that an LS-100 number has been reported by
the original authors.

## Result provenance

Every number is tagged with exactly one provenance label:

- `paper-reported`: copied from the source paper and shown only in related-work/supplementary context;
- `official-reproduced`: produced by unmodified official code on an original supported dataset;
- `official-component-transfer`: official decision/update components evaluated with the common
  frozen acoustic features on LS-100, NS-100, and FSC-89;
- `paper-reimplementation`: our paper-faithful implementation when official code is unavailable.

The main quantitative tables contain only three-dataset `official-transfer` and
`reimplementation` results under the fixed protocol. Failed or negative baselines are
retained rather than silently removed.

## Required experiment matrix

Every B0-B6 method and every ablation of our method must produce one result bundle on each
of LS-100, NS-100, and FSC-89. A method is not marked complete until all three bundles exist.
Hyperparameters copied from a source paper remain fixed where possible; any dataset-specific
selection uses validation data only and is reported in the appendix.

## Paper table schema

| Method | Year/Venue | Code | Replay | Updates backbone | Known ACC | Novel ACC | AUROC | AA-inc | Final-inc | AA-all | PD |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|

For each entry, save the command, commit hash, seed list, per-session raw values, mean,
standard deviation, runtime, and peak memory. Statistical comparison with our final method
uses paired seeds and reports both absolute gain and uncertainty.

## Audit notes

- The completed LS-100 runs tagged `teen_a09_dual05_10runs` and
  `teen_official_transfer_10runs` include the DFSB structure-discovery branch and therefore
  belong to a TEEN+DFSB ablation, not the pure TEEN baseline row.
- Pure TEEN is rerun on all three datasets with `structure_discovery_weight=0.0` via
  `scripts/run_teen_three_datasets.sh`.
- VB-CGCD official code assumes an incremental batch large enough for a fixed batch size of
  128 (the paper loader uses hundreds of novel samples per class). The LS/NS/FSC transfer
  uses `min(128, stream_size)` and skips S-learning when F-learning predicts zero novel
  candidates. Both are input-size guards: no oracle labels or substitute samples are used.
- VB-CGCD reports current-session novel accuracy, while this project reports cumulative
  incremental accuracy. The transfer runner stores both fields and never treats one as the
  other.
- The OFCL transfer retains its anchor bank, per-class metric-ball radius, open rejection,
  clustering, and anchor expansion. Its ViT prompt encoder is replaced by the shared frozen
  acoustic encoder, so these values are tagged `official-component-transfer`, not exact
  official reproduction. The first-seed results are stored in
  `experiments/ofcl_three_datasets.csv`; paired repeated streams remain required.
- Pure TEEN on the canonical LS-100 checkpoint completed at AA-inc 69.26% and final-inc
  55.96% ± 2.98%. The earlier LS checkpoint run remains audit-only.

`official-transfer` means the official algorithm is retained but the published image
backbone/dataset is replaced by the common LS-100 acoustic backbone and stream. Paper-only
reimplementations are never labelled as official results.

## Our method policy

Our final method is evaluated only after B0-B6. Modules are introduced one at a time:
structure-aware discovery, uncertainty/covariance memory, hardness-aware statistical replay,
and old-new boundary balancing. No baseline-specific oracle labels or test-set tuning are used.
