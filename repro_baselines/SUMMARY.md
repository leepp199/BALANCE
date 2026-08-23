# Baseline Reproduction Summary

## 1. Overview

This document summarizes the reproduction and comparison of baseline methods for **Few-Shot Open-World Audio Classification (FOWAC)**. All methods follow **Protocol A (Shared Encoder)** — using the same ResNet18-based MYNET audio encoder — to ensure fair comparison.

### 1.1 Datasets

| Dataset | Base Classes | Novel Classes | Total | Data Path |
|---------|:-----------:|:------------:|:-----:|-----------|
| LS-100 (LibriSpeech) | 80 | 20 | 100 | `/data/datasets/librispeech_fscil/` |
| NS-100 (NSynth) | 80 | 20 | 100 | `/data/datasets/The_NSynth_Dataset/nsynth-100-fs-meta/` |
| FSC-89 (FSD-MIX-CLIPS) | 69 | 20 | 89 | `/data/datasets/FSD-MIX-CLIPS-for_FSCIL/` |

### 1.2 Methods

**CIL Methods (7)**:

| Method | Source | Description |
|--------|--------|-------------|
| PITEL-CUSC | TASLP 2025, [Official Code](https://github.com/kaist-ina/PITEL-CUSC) | Pseudo-Incremental Stochastic Classifier |
| Fully-FCAC (MAR) | 2025, [Official Code](https://github.com/YongjieSi/MAR) | Ridge Regression Classifier |
| Tri-WE | CVPR 2025 | Tripartite Weight-Space Ensemble |
| MACIL | ICML 2025, [Official Code](https://github.com/conditionWang/MACIL) | Mean-Shift Compensation |
| CEC | CVPR 2021, [Official Code](https://github.com/icoz69/CEC-CVPR2021) | Graph Attention Prototype Evolution |
| PAN | NeurIPS 2021 | Prototype Augmentation Network |
| ProtoNet | NeurIPS 2017 | Prototypical Networks |

**OSR Methods (7)**:

| Method | Source | Description |
|--------|--------|-------------|
| MLS | ICLR 2022 | Maximum Logit Score |
| TANE | ICLR 2023 | Test-time Aggregation for OSR |
| Energy | NeurIPS 2020 | Energy-based OOD Detection |
| COSTARR | ICCV 2025 | Attenuation Hypothesis OSR |
| UTL | ICCV 2025 | U²WO + ULS Unknown Text Learning |
| FOAC-AIFP | TASLP 2026, [Official Code](https://github.com/Jiahao-123/FOAC) | Audio Few-shot Open-set |
| OAFN | KBS 2025 | Dual-Channel Calibration + Confidence Gap |

---

## 2. Complexity Measurement Results

**Protocol**: Only the inference path (`encode()`) is measured for MACs, reflecting real deployment cost. CIL head MACs computed analytically (cosine logit = `2 × dim × n_classes`).

### 2.1 Encoder Complexity (Shared)

| Metric | Value |
|--------|-------|
| Encoder MACs | 0.595 GMacs |
| Encoder Params | 11.18 M |
| AIT (avg) | 2.77–9.00 ms |
| Feature Dim | 512 |

AIT varies due to CUDA init overhead; all methods use the same ResNet18 encoder.

### 2.2 CIL Head Complexity

All CIL methods use the same cosine classification head for the Protocol A evaluation. The head MACs are negligible compared to the encoder.

| Metric | Value |
|--------|-------|
| CIL Head MACs (80 classes) | 8.19 × 10⁴ |
| CIL Head Params (80 classes × 512) | 0.041 M |

### 2.3 OSR Score Complexity

| Method | Score MACs | Learned Params | Notes |
|--------|:----------:|:--------------:|-------|
| MLS | 8.19 × 10⁴ | 0 | Basic max logit |
| Energy | 8.19 × 10⁴ | 0 | Energy score |
| TANE | 1.23 × 10⁵ | 0 | Energy + logsumexp |
| COSTARR | 1.64 × 10⁵ | 0 | Pre/post attenuation |
| UTL | 1.64 × 10⁵ | 0.0041 M | Basis vectors + combiner |
| FOAC-AIFP | 1.64 × 10⁵ | 1.576 M | OpenSetGenerator |
| OAFN | 9.01 × 10⁴ | 0 | Distance-weighted scoring |

### 2.4 Key Observation

All OSR methods add negligible computational overhead (< 0.0003 GMacs, i.e., < 0.05% of encoder). The encoder dominates total MACs. FOAC-AIFP has the largest parameter overhead (1.58M additional params from its prototype generator).

---

## 3. Baseline Comparison Experiments

### 3.1 Experiment Design (per baseline_compare.md §9)

**Group 1: Classic sanity baselines** (interpetable lower/middle baselines)

| # | CIL | OSR | Category |
|---|-----|-----|----------|
| 1 | ProtoNet | MLS | Basic combination |
| 2 | CEC | MLS | Graph-based CIL |
| 3 | CEC | TANE | CIL + energy-based OSR |
| 4 | PAN | MLS | Prototype augment CIL |

**Group 2: Strong 2025–2026 combined baselines**

| # | CIL | OSR | Rationale |
|---|-----|-----|-----------|
| 1 | PITEL-CUSC | COSTARR | Audio CIL + attenuation OSR |
| 2 | PITEL-CUSC | FOAC-AIFP | Audio CIL + audio OSR |
| 3 | Fully-FCAC | COSTARR | Ridge reg + attenuation OSR |
| 4 | Fully-FCAC | FOAC-AIFP | Ridge reg + audio OSR |
| 5 | Tri-WE | COSTARR | Weight ensemble + OSR |
| 6 | Tri-WE | UTL | Weight ensemble + unknown learning |
| 7 | MACIL | COSTARR | Drift correction + OSR |
| 8 | MACIL | FOAC-AIFP | Drift correction + audio OSR |

### 3.2 Evaluation Protocol

- **5 Sessions**: Session 0 (base training, 80/69 classes), Sessions 1–4 (incremental, 5 novel classes each)
- **Metrics per session**: All-class accuracy, Incremental accuracy, AUROC, FPR95
- **Aggregated Metrics**: AA_all (average all-class acc), AA_inc (average incremental acc), PD_all (performance drop)
- **Evaluation repetitions**: 50 test episodes per session, results averaged
- **Encoder**: Shared MYNET (ResNet18) with audio spectrogram frontend

### 3.3 Running Status

Experiments are currently running on LS-100 (GPU 0). Status of each experiment:

| Experiment | Status |
|------------|--------|
| ProtoNet × MLS | 🔄 RUNNING (base training epoch ~30) |
| CEC × MLS | ⏳ PENDING |
| CEC × TANE | ⏳ PENDING |
| PAN × MLS | ⏳ PENDING |
| PITEL-CUSC × COSTARR | ⏳ PENDING |
| PITEL-CUSC × FOAC-AIFP | ⏳ PENDING |
| Fully-FCAC × COSTARR | ⏳ PENDING |
| Fully-FCAC × FOAC-AIFP | ⏳ PENDING |
| Tri-WE × COSTARR | ⏳ PENDING |
| Tri-WE × UTL | ⏳ PENDING |
| MACIL × COSTARR | ⏳ PENDING |
| MACIL × FOAC-AIFP | ⏳ PENDING |

### 3.4 Expected Output

After all experiments complete, results will be available in: `repro_baselines/results/summary.csv`

Schema: `dataset,cil,osr,S0_all,S1_all,S2_all,S3_all,S4_all,AA_all,AA_inc,PD_all,AUROC_S0,AUROC_S4,FPR95_S0,FPR95_S4`

---

## 4. Implementation Notes

### 4.1 Official Source Code Used

| Method | Repository | Key Algorithm |
|--------|-----------|---------------|
| PITEL-CUSC | `official_code/PITEL-CUSC` | Stochastic classifier with Softplus(σ-4) noise |
| Fully-FCAC (MAR) | `official_code/MAR` | Ridge regression `W = solve(G+λI, Q)ᵀ` |
| MACIL | `official_code/MACIL` | RBF-weighted mean-shift compensation |
| FOAC-AIFP | `official_code/FOAC-AIFP` | OpenSetGenerator from base prototypes |
| CEC | `official_code/CEC-CVPR2021` | Multi-head graph attention |

### 4.2 Paper-Based Implementations

| Method | Paper | Adapted Algorithm |
|--------|-------|-------------------|
| Tri-WE | CVPR 2025, arXiv:2506.15720 | 3-way weight interpolation + ADKD |
| UTL | ICCV 2025 | U²WO basis vectors + ULS contrastive loss |
| OAFN | KBS 2025 | Dual-channel calibration + confidence gap |
| COSTARR | ICCV 2025 | Attenuation-based consolidation |
| ProtoNet | NeurIPS 2017 | Cosine prototype classification |
| PAN | NeurIPS 2021 | Prototype augmentation |
| MLS/TANE/Energy | ICLR 2022/2023, NeurIPS 2020 | Logit-based OSR scores |

### 4.3 Repository Structure

```
repro_baselines/
├── methods/
│   ├── base.py           # Shared CILBase/OSRBase interfaces
│   ├── cil/              # 7 CIL methods
│   │   ├── pitel_cusc.py
│   │   ├── fully_fcac.py
│   │   ├── triwe.py
│   │   ├── macil.py
│   │   ├── cec.py
│   │   ├── pan.py
│   │   └── prototypical.py
│   └── osr/              # 7 OSR methods
│       ├── mls.py
│       ├── tane.py
│       ├── energy.py
│       ├── costarr.py
│       ├── utl.py
│       ├── foac_aifp.py
│       └── oafn.py
├── measure_complexity.py  # Complexity measurement
├── train.py               # 5-session training & evaluation
├── run_experiments.sh     # Batch experiment runner (Group 1 + Group 2)
├── run_pipeline.sh        # Full pipeline (all 147 combinations)
├── official_code/         # Cloned official repositories
├── results/               # Complexity results
├── logs/                  # Training logs
└── checkpoints/           # Saved model weights
```

---

## 5. Report Generation

This document was auto-generated on 2026-04-26. To regenerate:

```bash
# 1. Run complexity measurement (already done)
# 2. Run comparison experiments (in progress)
bash repro_baselines/run_experiments.sh

# 3. After all experiments complete, view summary
cat repro_baselines/results/summary.csv

# 4. Full pipeline (all 147 combinations)
bash repro_baselines/run_pipeline.sh
```
