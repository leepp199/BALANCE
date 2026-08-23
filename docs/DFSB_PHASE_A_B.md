# DFSB Phase A–B implementation record

## Verified forward path

`waveform [B,L] → Spectrogram → LogmelFilterBank → bn0 → repeat to 3 channels →
ResNet conv1/maxpool/layer1/layer2/layer3/layer4 → [B,512,H,W] → adaptive average
pooling → semantic embedding [B,512] → fc`

`ResNet.forward()` returns the unpooled layer4 map. Pooling is performed by
`MYNET.encode()` / `MYNET.base_encode()`. DFSB uses `MYNET.forward_to_layer4()` and
therefore does not change the baseline backbone, input size, or pooling.

## Legacy CFFM location

The active legacy module is not a 512-stage module in the main encoder forward.
`MYNET.enhance_encode()` inserts `LocalFeatureCluster` after `layer1`; optional OSR
evaluation calls this path through `--use_cffm_eval`. Unknown clustering also contains
a separate cached/local clustering path. The standard `encode()` path does not invoke
legacy CFFM.

## Phase B pipeline

1. `extract_deep_features.py`: stream the complete base-training split through the
   warm-up checkpoint and maintain a bounded uniform random-key reservoir of normalized
   layer4 descriptors. GPU memory holds only one batch; the CPU reservoir is bounded.
2. `fit_structure_bank.py`: fit one global `MiniBatchKMeans` model to the sampled
   cross-sample descriptor matrix and save normalized centers to `structure_bank.pt`.
3. `assign_structure_targets.py`: compute hard assignments `[B,H,W]`, structural
   responses `[B,K]`, and quantization residual `[B]` without fitting at inference.
4. `validate_structure_bank.py`: verify shapes, finiteness, response normalization, and
   that the same loaded bank instance is applied to multiple samples.

The centers are shared deep-feature reference points. They are not semantic classes or
acoustic units, and K is independent of the classification class count C.

## Real LibriSpeech verification (seed 3420)

- Warm-up checkpoint: `/data/lqq/baseline/save/base_train_for_meta_ls.pth`.
- Base split: 40,000 samples.
- Measured layer4 shape: `[B,512,7,4]`; 28 descriptors per audio.
- Descriptors observed: 1,120,000; uniformly retained: 200,000.
- Global MiniBatchKMeans: K=128, input `[200000,512]`, output bank `[128,512]`.
- Validation batch: assignments `[128,7,4]`, response `[128,128]`, residual `[128]`.
- Four-batch preview (512 samples): residual mean/std `0.16970/0.02730`, range
  `[0.10774,0.25622]`; every response was finite and summed to one.

## Existing execution entries

- Base training: `train_unopenset.py:main → train → base_train →
  standard_base_train_with_metrics`; effective objective is classification CE plus the
  optional uncertainty center loss.
- Open-set rejection: `train_unopenset.py → threshold_free.run_test_fsl`; the default
  path uses pooled `hgnn_encode` features and positive/negative prototype margins.
- Unknown clustering and expansion: `train_unopenset.debug_cluster`; K-means groups
  rejected embeddings and writes new means into `model.fc.weight`.
- Incremental evaluation: `get_testloader/get_inc_testloader → test`, after prototype
  expansion and before advancing `num_labeled_classes`.

## Phase C insertion plan

Keep the warm-up checkpoint fixed, load `structure_bank.pt`, and add a lightweight
masked structure predictor on the unpooled layer4 map. Hard assignments from the frozen
bank are stop-gradient targets; cross-entropy is evaluated only at masked latent
positions. The training objective becomes `L_cls + existing optional center term +
lambda_str L_str`, guarded by `feature_structure_mode` and `structure_prediction`
switches. The baseline pooling and classifier path remain unchanged.
