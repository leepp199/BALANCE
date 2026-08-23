# Meta-training audit for open-set recognition

## What the original code actually optimizes

`models/metatrainer_oo.py::train_episode` receives support/query episodes and an
open-set episode. The model returns three losses: closed-set classification,
open-set hinge, and FUNIT. The optimized objective is

`L = L_cls + gamma * L_open_hinge + funit * L_funit`.

The uncertainty curriculum does retain the nuclear-norm idea, but applies it to the
centered MC prediction matrix. In this form the norm measures disagreement between
stochastic predictions. The uncentered norm is logged as a confidence/dispersity
diagnostic and is not interpreted monotonically as difficulty.

The implementation also computes an episode AUROC from `negative_score -
positive_score` for known queries versus open-set queries. Therefore the original
meta-training is not purely closed-set training: it does expose the negative
prototype generator and open-set boundary to gradients.

## Problems that prevent a strong claim

1. `train()` calls `meta_train(..., eval_loader=None)`, so the apparent meta-test
   branch is never used during the normal experiment.
2. The checkpoint-selection code, when an evaluator is supplied, tracks training
   episode accuracy/AUROC rather than a held-out open-set AUROC. This can select a
   classifier that fits episodes but has a weak unknown boundary.
3. The default encoder is frozen in meta-training. Thus open-set adaptation mostly
   changes classifier/negative-prototype parameters and cannot repair a dataset-level
   representation whose known and unknown classes overlap.
4. The final incremental evaluator uses a separate thresholding/calibration path;
   meta-training AUROC is not automatically calibrated to the mixed-stream
   distribution used at test time.

## Required ablation for the paper

For each dataset, use the same retrained base checkpoint and paired streams for:

* no meta-training (`skip_meta_train=True`);
* original meta-training (frozen encoder, current objective);
* OSR-aware meta-training with a held-out open-set validation episode for model
  selection and explicit AUROC/FPR95 logging;
* optional low-LR layer-4 fine-tuning with a base anchor, reported only if it
  improves both base accuracy and open-set AUROC.

The paper should claim meta-training is useful only when the second/third rows
improve held-out AUROC/F1 without degrading base accuracy. Training-episode AUROC
alone is diagnostic, not evidence.
