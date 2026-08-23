# Reviewer-driven revision matrix

Source: Interspeech 2026 reviews for paper 1011. This file is an internal revision
checklist; it is not a rebuttal and does not copy review text into the manuscript.

| Reviewer concern | Revision action | Required evidence | Status |
|---|---|---|---|
| PUPG implementation and its incremental boundary are unclear | Add explicit support/weight definitions, generator inputs/outputs, loss, and the positive-vs-negative prototype decision used at every session | Algorithm box, equations, tensor dimensions, threshold diagnostics | In progress |
| Time/frequency Euclidean interpretation is technically questionable | Remove the physical time/frequency claim; replace sample-wise CFFM with cross-sample DFSB over latent layer-4 descriptors | DFSB bank metadata, response/residual equations, semantic-only vs structural ablation | Implemented in code; experiments incomplete |
| CFFM “sparse feature mining” and Eq. 9 lack meaning | Delete the unsupported sparse-mining claim and old positional-cluster equation; define shared structural response and quantization residual instead | New method text and figure | Unsupported claim removed; replacement equations drafted; figure pending |
| Missing training/backbone/classifier configuration | Report input preprocessing, ResNet-18 stages, layer-4 tensor shape, pooling, 512-D embedding, cosine classifier, optimizer, schedule, task construction, seeds and hardware | Reproducibility table | Drafted from executable configs; version/checkpoint hash manifest complete |
| NS-100 ablations missing | Run every principal ablation on LS-100, NS-100 and FSC-89 | Three-dataset ablation table with mean/std | Running |
| Baselines are old | Add recent TEEN, VB-CGCD, OFCL, OPCR, YLOC and reviewer-named audio methods where protocol-compatible; distinguish exact, official-transfer and reimplementation | Provenance ledger and three-dataset results | Partly complete |
| Reviewer specifically names PCLAE-CTPN, OAFN, FFCAC and DNPG | Verify task compatibility; reproduce compatible components or explain protocol mismatch. OAFN is noisy episodic FSL rather than continual discovery; FFCAC assumes few-shot base data; DNPG is FSOR only | Compatibility table plus combined OSR/CIL transfer where defensible | Research started |
| Only LS-100/NS-100 and controlled protocol | Add FSC-89 environmental sounds; report mixed known/novel stream; include harder unknown/domain/noise study if feasible | FSC-89 table and robustness subsection | FSC baseline weakness identified; robustness pending |
| Evaluation lacks class-specific and stability metrics | Add known accuracy, unknown recall, F1, AUROC, AUPR, FPR95, clustering ACC/NMI/ARI, old/new accuracy, AA, final accuracy, PD/forgetting, per-session curves and confusion analysis | Raw per-seed records and figures | Partly implemented |
| Limited ablation and unclear dual encoder choice | Compare baseline, old CFFM, DFSB base only, structure-aware OSR, joint clustering, dual prototype, memory, replay and boundary modules | Factorial ablation and sensitivity plots | Pending |
| Prior work/origins not acknowledged | Rewrite Related Work by FCAC, FSOR, GCD/CGCD, FSCIL and audio open-world learning; cite each inherited component | Updated bibliography and attribution paragraph | Related-work draft and verified recent citations added; complete bibliography audit pending |
| Formatting/template/equation/reference errors | Move to ICASSP template, repair references, notation, operators, captions, tables and acknowledgement handling | Compilable final LaTeX/PDF | ICASSP-2027 four-page LaTeX source scaffolded; official 2027 kit and final compile pending |
| Figures hard to parse | Rebuild framework, PUPG, DFSB and continual-memory figures with consistent boundaries and variable labels | Vector/PDF figures and caption audit | Candidate-quality and continual-curve vectors complete; framework panels pending |

## Scope decisions

- OAFN addresses noisy episodic few-shot event classification, not continual novel-class
  discovery. It is relevant to robustness discussion and a noise stress test, but its reported
  numbers are not directly comparable to the 80+4x5 mixed-stream protocol.
- FFCAC changes the base-data regime to few-shot base classes. It should be cited and its
  classifier/embedding components may be transferred, but original-paper numbers cannot be
  inserted into the main table.
- DNPG provides an FSOR negative-prototype generator. It can replace PUPG in the common
  pipeline as an OSR-component baseline, after which the same clustering and expansion stage
  must be used for fair end-to-end evaluation.
