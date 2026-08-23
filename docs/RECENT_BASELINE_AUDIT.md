# Recent baseline audit (2024–2026)

This audit distinguishes protocol-compatible comparisons from adjacent tasks. A method is
eligible for the main table only if it can consume a mixed known/novel unlabeled stream,
expand the classifier continually, and report old/new/all performance without using extra
labels or modalities.

| Method | Venue/year | Official source | Compatibility decision | Planned evidence |
|---|---:|---|---|---|
| Happy | NeurIPS 2024 | https://github.com/mashijie1028/Happy-CGCD | High conceptual compatibility; image encoder training is not directly portable. Hardness-aware prototype sampling and entropy terms can be transferred to frozen audio features. | Add feature-transfer baseline on LS/NS/FSC and label it as a component reimplementation. |
| CAMP | ECCV 2024 | https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1766_ECCV_2024_paper.php | GCCD-compatible, but requires projected feature distillation and category-adaptation training. | Compatibility table unless an audio encoder training budget is completed. |
| PromptCCD | ECCV 2024 | Listed in the public Category Discovery index and official project links | Continual discovery with Gaussian-mixture prompt pools; assumes promptable visual foundation models. | Compatibility table; a Gaussian-mixture feature analogue may be evaluated separately. |
| VB-CGCD | ICML 2025 | https://proceedings.mlr.press/v267/dai25a.html and https://github.com/daihao42/VB-CGCD | Closest recent C-GCD baseline. | Official-code three-dataset transfer completed; strict 5-shot full-covariance collapse is reported as DNF with diagnostics, not silently omitted. |
| FaE | AAAI 2026 | https://ojs.aaai.org/index.php/AAAI/article/view/37530 | Continual GCD and highly relevant stability/plasticity design; publication is newer than the target paper's original review set. | Implement frozen-feature Fix/Explore analogue or list as post-review concurrent work depending on ICASSP cutoff. |
| Multi-view MMI-CGCD | ESWA 2025 | https://doi.org/10.1016/j.eswa.2024.125994 | Relevant C-GCD but needs class-level contrastive encoder training and multiple learned views. | Compatibility table unless full encoder training is run. |
| Fully-FCAC / FFCAC | Interspeech 2025 | https://arxiv.org/abs/2506.18406 | Audio and few-shot class incremental, but fully few-shot base training and labeled novel sessions differ from our unlabeled discovery stream. | Reviewer-requested adjacent-task table; not a main-table number. |
| Temporal prompting FS-AVCIL | AAAI 2025 | https://doi.org/10.1609/aaai.v39i15.33770 | Uses audio-visual inputs and labeled incremental examples. | Related work only; extra modality makes numerical comparison invalid. |

## Reporting rule

Official results, official-code transfers, and paper-inspired component reimplementations
must appear in separate provenance columns. A failed official transfer remains visible as
DNF with the failure cause. No image benchmark number is copied into an audio table.
