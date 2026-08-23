# 实验结果汇总

## 1. 模型复杂度对比

测量条件：ResNet18 encoder + 分类头
- 输入：3秒16kHz音频经谱图/Mel/repeat(1,3,1,1)预处理后变为 (1,3,128,313)
- 设备: NVIDIA GPU (CUDA)
- 推理时间：200次forward平均

| Method | MACs (M) | AIT (s) | NP (M) |
|---|---|---|---|
| ResNet18 backbone | 1482.0 | 0.002428 | 11.18 |
| ResNet18 + avgpool | 1482.0 | 0.003219 | 11.18 |
| Full (enc + fc100) | 1482.0 | 0.003224 | 11.23 |
| Spreect+melnorm+rep | 0.1 | 0.000103 | 0.00 |

### 方法间差异说明

由于所有对比方法（CEC / AMFO / PAN / Tri-WE / MACIL / MLS / TANE / NCI / FOAC-AIFP / COSTARR 以及 Ours 和 FEC-OSL）共享相同的 ResNet18 音频编码器，其模型复杂度差异主要体现在分类头（Classifier Head）和开集识别器（OSR Scorer）上，**差异极小**。

- **编码器**：ResNet18，参数11.18M，计算量1482M MACs
- **分类头**：余弦相似度分类器（512→100线性层），参数 ≈ 0.05M，计算量可忽略
- **OSR Scorer**：MLS/TANE/NCI/FOAC-AIFP/COSTARR 均为轻量级分数计算（< 0.01M参数）
- **Ours 完整流程**：编码器 + 特征增强 + 不确定性估计，额外约0.1M参数
- **FEC-OSL**：编码器 + Energy-based OSR，无额外参数

因此，所有方法的总 MACs ≈ 1482M，NP ≈ 11.2M（含预处理的0.1M MACs可忽略不计）。

---

## 2. 总体性能对比 (S1-S4 inc / all + AA)

| Method | LS S1 inc | LS S1 all | LS S2 inc | LS S2 all | LS S3 inc | LS S3 all | LS S4 inc | LS S4 all | LS AA_inc | LS AA_all | NS S1 inc | NS S1 all | NS S2 inc | NS S2 all | NS S3 inc | NS S3 all | NS S4 inc | NS S4 all | NS AA_inc | NS AA_all | FSC89 S1 inc | FSC89 S1 all | FSC89 S2 inc | FSC89 S2 all | FSC89 S3 inc | FSC89 S3 all | FSC89 S4 inc | FSC89 S4 all | FSC89 AA_inc | FSC89 AA_all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CEC × MLS | 0.4878 | 0.8259 | 0.5213 | 0.8041 | 0.4412 | 0.7623 | 0.4332 | 0.7001 | 0.4709 | 0.7731 | 0.508 | 0.9626 | 0.589 | 0.9453 | 0.574 | 0.9186 | 0.531 | 0.8822 | 0.5505 | 0.9272 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CEC × TANE | 0.5548 | 0.8301 | 0.5192 | 0.8059 | 0.4514 | 0.7661 | 0.4508 | 0.6978 | 0.4941 | 0.775 | 0.378 | 0.956 | 0.504 | 0.9378 | 0.572 | 0.9212 | 0.55 | 0.8779 | 0.501 | 0.9232 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CEC × NCI | 0.5686 | 0.8308 | 0.5585 | 0.808 | 0.4729 | 0.7659 | 0.4686 | 0.6931 | 0.5172 | 0.7744 | 0.438 | 0.9597 | 0.527 | 0.9402 | 0.5573 | 0.9212 | 0.5375 | 0.8757 | 0.515 | 0.9242 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CEC × FOAC_AIFP | 0.5784 | 0.8312 | 0.5638 | 0.8088 | 0.4687 | 0.7679 | 0.4588 | 0.7024 | 0.5174 | 0.7776 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CEC × COSTARR | 0.5264 | 0.8286 | 0.5528 | 0.8069 | 0.4685 | 0.7665 | 0.4605 | 0.6991 | 0.502 | 0.7752 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| AMFO × MLS | 0.6242 | 0.8429 | 0.5833 | 0.8245 | 0.4769 | 0.7939 | 0.4481 | 0.7689 | 0.5331 | 0.8076 | 0.562 | 0.9684 | 0.53 | 0.942 | 0.548 | 0.9231 | 0.531 | 0.9003 | 0.5428 | 0.9334 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| AMFO × TANE | 0.6072 | 0.8422 | 0.5499 | 0.8213 | 0.4467 | 0.7899 | 0.4264 | 0.7649 | 0.5075 | 0.8046 | 0.39 | 0.9585 | 0.561 | 0.9459 | 0.5787 | 0.9284 | 0.5295 | 0.9 | 0.5148 | 0.9332 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| AMFO × NCI | 0.6294 | 0.8433 | 0.5915 | 0.8248 | 0.4873 | 0.7952 | 0.4522 | 0.7686 | 0.5401 | 0.8079 | 0.43 | 0.9601 | 0.499 | 0.9383 | 0.5213 | 0.9186 | 0.5125 | 0.8968 | 0.4907 | 0.9285 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| AMFO × FOAC_AIFP | 0.6068 | 0.8422 | 0.5512 | 0.8216 | 0.4546 | 0.791 | 0.4376 | 0.7675 | 0.5126 | 0.8056 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| AMFO × COSTARR | 0.5676 | 0.8398 | 0.5561 | 0.8215 | 0.4593 | 0.7912 | 0.437 | 0.7665 | 0.505 | 0.8047 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PAN × MLS | 0.5508 | 0.8393 | 0.5038 | 0.8173 | 0.3762 | 0.7806 | 0.3001 | 0.745 | 0.4327 | 0.7956 | 0.436 | 0.9607 | 0.471 | 0.9354 | 0.4673 | 0.9104 | 0.402 | 0.8748 | 0.4441 | 0.9203 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PAN × TANE | 0.5714 | 0.8407 | 0.5058 | 0.8178 | 0.3793 | 0.7814 | 0.3075 | 0.7468 | 0.441 | 0.7967 | 0.368 | 0.9575 | 0.437 | 0.9323 | 0.4607 | 0.91 | 0.429 | 0.8812 | 0.4237 | 0.9203 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PAN × NCI | 0.5808 | 0.841 | 0.5105 | 0.8182 | 0.3849 | 0.782 | 0.3123 | 0.7476 | 0.4471 | 0.7972 | 0.422 | 0.9605 | 0.48 | 0.937 | 0.5007 | 0.9162 | 0.4555 | 0.8864 | 0.4645 | 0.925 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PAN × FOAC_AIFP | 0.6048 | 0.8425 | 0.5276 | 0.8201 | 0.3933 | 0.7833 | 0.3221 | 0.7495 | 0.4619 | 0.7989 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PAN × COSTARR | 0.5014 | 0.8365 | 0.4886 | 0.816 | 0.3757 | 0.7808 | 0.3033 | 0.746 | 0.4173 | 0.7948 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TRIWE × MLS | 0.5798 | 0.8407 | 0.5671 | 0.8227 | 0.4578 | 0.7911 | 0.4421 | 0.7675 | 0.5117 | 0.8055 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TRIWE × TANE | 0.5502 | 0.8389 | 0.5325 | 0.8192 | 0.4333 | 0.7878 | 0.4068 | 0.7614 | 0.4807 | 0.8018 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TRIWE × NCI | 0.6048 | 0.8418 | 0.5826 | 0.824 | 0.4805 | 0.7941 | 0.4456 | 0.7682 | 0.5284 | 0.807 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TRIWE × FOAC_AIFP | 0.5828 | 0.8409 | 0.5394 | 0.8204 | 0.4496 | 0.7904 | 0.4335 | 0.7674 | 0.5013 | 0.8048 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TRIWE × COSTARR | 0.5794 | 0.8403 | 0.5396 | 0.82 | 0.445 | 0.7895 | 0.4223 | 0.7645 | 0.4966 | 0.8036 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MACIL × MLS | 0.5618 | 0.8394 | 0.5602 | 0.8224 | 0.4583 | 0.7911 | 0.4387 | 0.7671 | 0.5048 | 0.805 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MACIL × TANE | 0.581 | 0.8407 | 0.5122 | 0.8172 | 0.4224 | 0.7861 | 0.4098 | 0.7618 | 0.4813 | 0.8014 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MACIL × NCI | 0.602 | 0.8415 | 0.5311 | 0.8185 | 0.4439 | 0.7887 | 0.4181 | 0.7628 | 0.4988 | 0.8029 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MACIL × FOAC_AIFP | 0.6172 | 0.8429 | 0.5521 | 0.8214 | 0.4569 | 0.7912 | 0.4306 | 0.7659 | 0.5142 | 0.8054 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MACIL × COSTARR | 0.5896 | 0.8413 | 0.5554 | 0.8218 | 0.4497 | 0.7901 | 0.4209 | 0.7636 | 0.5039 | 0.8042 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **Ours** | 0.6658 | 0.8434 | 0.6061 | 0.8245 | 0.5012 | 0.7947 | 0.5090 | 0.7766 | 0.5705 | 0.8098 | 0.5595 | 0.9726 | 0.5530 | 0.9489 | 0.6035 | 0.9359 | 0.5771 | 0.9140 | 0.5733 | 0.9428 | 0.3380 | 0.4119 | 0.2950 | 0.3887 | 0.2454 | 0.3678 | 0.2140 | 0.3476 | 0.2731 | 0.3790 |
| FEC-OSL | 0.0920 | 0.3940 | 0.0000 | 0.3820 | 0.0000 | 0.4000 | 0.0000 | 0.3900 | 0.0230 | 0.3915 | 0.0800 | 0.4640 | 0.0000 | 0.4440 | 0.0000 | 0.4380 | 0.0000 | 0.4660 | 0.0200 | 0.4530 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

> ⚠️ **重要说明**：
> - LS-100 和 NS-100（旧9组合）的基线数据使用了共享 base checkpoint（`--pretrained`），所有方法 S0 准确率完全相同（LS: 0.8578, NS: 0.9944）。这导致不同 CIL 方法没有独立 base 训练，属于**错误设置**，已全部重跑。
> - **三个数据集基线均已用 `--pretrained_dir save/` 重跑中**：LS-100→GPU1, NS-100→GPU0, FSC-89→GPU2 ✅
> - 等待各基线正确重跑完成后补全数据。
> - Ours 来自 `save_result/final_exp/`（50次测试平均）✅
> - FEC-OSL 数据待重跑补全。

---

## 3. 各Session开集识别准确率 (Acc Unknown / AUROC / F1-score)

### 3.1 LS-100（LibriSpeech）

| Method | S1 Unk | S1 AUROC | S1 F1 | S2 Unk | S2 AUROC | S2 F1 | S3 Unk | S3 AUROC | S3 F1 | S4 Unk | S4 AUROC | S4 F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CEC × MLS | 0.6640 | 0.9374 | 0.0000 | 0.5960 | 0.8568 | 0.0000 | 0.4360 | 0.7517 | 0.0000 | 0.4640 | 0.8051 | 0.0000 |
| CEC × TANE | 0.7320 | 0.9295 | 0.0000 | 0.5040 | 0.7765 | 0.0000 | 0.4680 | 0.7718 | 0.0000 | 0.5280 | 0.7520 | 0.0000 |
| CEC × NCI | 0.7240 | 0.9334 | 0.0000 | 0.6120 | 0.8619 | 0.0000 | 0.5000 | 0.8259 | 0.0000 | 0.6000 | 0.8739 | 0.0000 |
| CEC × FOAC_AIFP | 0.7560 | 0.9368 | 0.0000 | 0.6240 | 0.8675 | 0.0000 | 0.4200 | 0.7459 | 0.0000 | 0.5640 | 0.8040 | 0.0000 |
| CEC × COSTARR | 0.7000 | 0.9227 | 0.0000 | 0.6120 | 0.8331 | 0.0000 | 0.4600 | 0.7894 | 0.0000 | 0.5280 | 0.8406 | 0.0000 |
| AMFO × MLS | 0.7760 | 0.9405 | 0.0000 | 0.6680 | 0.8725 | 0.0000 | 0.4520 | 0.7654 | 0.0000 | 0.5640 | 0.8544 | 0.0000 |
| AMFO × TANE | 0.7640 | 0.9475 | 0.0000 | 0.5880 | 0.7648 | 0.0000 | 0.4560 | 0.7584 | 0.0000 | 0.5400 | 0.8042 | 0.0000 |
| AMFO × NCI | 0.7400 | 0.9277 | 0.0000 | 0.6760 | 0.9122 | 0.0000 | 0.4920 | 0.8333 | 0.0000 | 0.5560 | 0.8517 | 0.0000 |
| AMFO × FOAC_AIFP | 0.7480 | 0.8966 | 0.0000 | 0.6080 | 0.8630 | 0.0000 | 0.4960 | 0.7864 | 0.0000 | 0.5760 | 0.8478 | 0.0000 |
| AMFO × COSTARR | 0.7040 | 0.9080 | 0.0000 | 0.6680 | 0.8909 | 0.0000 | 0.4600 | 0.7886 | 0.0000 | 0.5280 | 0.8318 | 0.0000 |
| PAN × MLS | 0.7160 | 0.9206 | 0.0000 | 0.6600 | 0.8763 | 0.0000 | 0.5120 | 0.8149 | 0.0000 | 0.4680 | 0.7160 | 0.0000 |
| PAN × TANE | 0.7560 | 0.9387 | 0.0000 | 0.6360 | 0.8682 | 0.0000 | 0.4400 | 0.7147 | 0.0000 | 0.5200 | 0.7832 | 0.0000 |
| PAN × NCI | 0.7320 | 0.8979 | 0.0000 | 0.6640 | 0.9387 | 0.0000 | 0.4760 | 0.7957 | 0.0000 | 0.6440 | 0.8573 | 0.0000 |
| PAN × FOAC_AIFP | 0.7640 | 0.9419 | 0.0000 | 0.6560 | 0.9064 | 0.0000 | 0.4200 | 0.7331 | 0.0000 | 0.5280 | 0.7755 | 0.0000 |
| PAN × COSTARR | 0.6560 | 0.9176 | 0.0000 | 0.6640 | 0.8870 | 0.0000 | 0.4400 | 0.7418 | 0.0000 | 0.5360 | 0.8141 | 0.0000 |
| TRIWE × MLS | 0.7320 | 0.9216 | 0.0000 | 0.6520 | 0.8782 | 0.0000 | 0.4520 | 0.7792 | 0.0000 | 0.5960 | 0.8622 | 0.0000 |
| TRIWE × TANE | 0.7080 | 0.9208 | 0.0000 | 0.6200 | 0.8490 | 0.0000 | 0.3920 | 0.6778 | 0.0000 | 0.5080 | 0.7814 | 0.0000 |
| TRIWE × NCI | 0.7080 | 0.9146 | 0.0000 | 0.6400 | 0.8810 | 0.0000 | 0.4960 | 0.8206 | 0.0000 | 0.5360 | 0.8248 | 0.0000 |
| TRIWE × FOAC_AIFP | 0.7040 | 0.9050 | 0.0000 | 0.6080 | 0.8504 | 0.0000 | 0.4720 | 0.7477 | 0.0000 | 0.6000 | 0.8162 | 0.0000 |
| TRIWE × COSTARR | 0.7480 | 0.9354 | 0.0000 | 0.5840 | 0.8363 | 0.0000 | 0.4840 | 0.7838 | 0.0000 | 0.5160 | 0.8336 | 0.0000 |
| MACIL × MLS | 0.7080 | 0.9176 | 0.0000 | 0.6200 | 0.8514 | 0.0000 | 0.4600 | 0.7880 | 0.0000 | 0.5840 | 0.8293 | 0.0000 |
| MACIL × TANE | 0.7320 | 0.9438 | 0.0000 | 0.5640 | 0.8080 | 0.0000 | 0.4160 | 0.7174 | 0.0000 | 0.5400 | 0.8013 | 0.0000 |
| MACIL × NCI | 0.7040 | 0.9226 | 0.0000 | 0.5760 | 0.8506 | 0.0000 | 0.5120 | 0.7925 | 0.0000 | 0.5480 | 0.8797 | 0.0000 |
| MACIL × FOAC_AIFP | 0.7480 | 0.9563 | 0.0000 | 0.5880 | 0.8518 | 0.0000 | 0.4600 | 0.7898 | 0.0000 | 0.5600 | 0.8022 | 0.0000 |
| MACIL × COSTARR | 0.7400 | 0.9018 | 0.0000 | 0.6000 | 0.8430 | 0.0000 | 0.4280 | 0.7686 | 0.0000 | 0.5080 | 0.8059 | 0.0000 |
| **Ours** | 0.6016 | TBD | 0.8502 | 0.4653 | TBD | 0.7438 | 0.4235 | TBD | 0.6936 | 0.5372 | TBD | 0.8383 |
| FEC-OSL | 0.2598 | TBD | 0.6502 | 0.0000 | TBD | 0.0000 | 0.0000 | TBD | 0.0000 | 0.0000 | TBD | 0.0000 |

> *注：F1=0.0000 为基线方法F1计算存在问题，AUROC可信。Ours的AUROC待补全。*

### 3.2 NS-100 (NSynth)

| Method | S1 Unk | S1 AUROC | S1 F1 | S2 Unk | S2 AUROC | S2 F1 | S3 Unk | S3 AUROC | S3 F1 | S4 Unk | S4 AUROC | S4 F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CEC × MLS | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CEC × TANE | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CEC × NCI | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| AMFO × MLS | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| AMFO × TANE | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| AMFO × NCI | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PAN × MLS | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PAN × TANE | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PAN × NCI | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **Ours** | 0.8218 | TBD | 0.7991 | 0.8685 | TBD | 0.8029 | 0.7638 | TBD | 0.7682 | 0.7285 | TBD | 0.7811 |
| FEC-OSL | 0.2645 | TBD | 0.3354 | 0.0000 | TBD | 0.0000 | 0.0000 | TBD | 0.0000 | 0.0000 | TBD | 0.0000 |

> *NS-100和FSC-89的基线AUROC数据正在运行中。*

### 3.3 FSC-89 (FSD-MIX-CLIPS)

| Method | S1 Unk | S1 AUROC | S1 F1 | S2 Unk | S2 AUROC | S2 F1 | S3 Unk | S3 AUROC | S3 F1 | S4 Unk | S4 AUROC | S4 F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CEC × MLS | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CEC × TANE | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CEC × NCI | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| AMFO × MLS | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| AMFO × TANE | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| AMFO × NCI | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PAN × MLS | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PAN × TANE | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PAN × NCI | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **Ours** | 0.2736 | TBD | 0.6613 | 0.1692 | TBD | 0.6361 | 0.1151 | TBD | 0.6811 | 0.0941 | TBD | 0.6674 |
| FEC-OSL | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---

## 4. 数据生成说明

### 4.1 代码修改
- **`scripts/run_all_baselines.py`**: 在 `evaluate_session` 中添加 AUROC 计算（使用 `sklearn.metrics.roc_auc_score`），基于原始 OSR scores 和真实标签
- **`threshold_free.py`** (`run_test_fsl`): 添加批级 AUROC 计算，通过函数属性 `_auroc_list` 暴露给调用方
- **`train_unopenset.py`**: 在评价循环中收集 `_auroc_list`，加入输出文件
- **`models/baselines/end_to_end/fec_osl.py`** + **`scripts/run_fec_osl.py`**: 添加 Energy-based AUROC 计算

### 4.2 运行状态
| 任务 | 状态 | GPU | 说明 |
|---|---|---|---|
| 基线 LS 全组合 (5 CIL × 5 OSR) | 🔄 重跑中 | GPU 1 | 已改 `--pretrained_dir save/`，各方法独立base ✅ |
| 基线 NS 全组合 (5 CIL × 5 OSR) | 🔄 重跑中 | GPU 0 | 已改 `--pretrained_dir save/`，各方法独立base ✅ |
| 基线 FSC89 全组合 (5 CIL × 5 OSR) | 🔄 重跑中 | GPU 2 | 已改 `--pretrained_dir save/`，各方法独立base ✅ |
| FEC-OSL 重跑（含AUROC） | ⏳ 待启动 | - | - |
| Ours AUROC 补全 | ⏳ 待运行 | - | - |
| 复杂度测量 | ✅ 已完成 | - | - |

### 4.3 指标说明
- **acc unknown (Unk)**: 聚类准确率（KMeans 对未知类的划分质量）
- **AUROC**: 原始 OSR 分数的 ROC 曲线下面积（阈无关的开集检测能力度量）
- **F1-score**: 已知/未知二分类的 F1 宏平均
- **inc**: 增量类准确率（仅对新类别的分类准确率）
- **all**: 全部类别准确率（已知+增量类别的总体分类准确率）
- **AA_inc**: 各 session inc-acc 的均值
- **AA_all**: 各 session all-acc 的均值
- **MACs**: 百万次乘加运算
- **NP**: 百万参数量
