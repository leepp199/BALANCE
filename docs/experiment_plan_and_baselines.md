# 对比实验执行清单（按任务重构）

## 1. 方法组

- G1 FSCIL-only：TOPIC / CEC / FACT
- G2 OSR-only：OpenMax / PROSER / ARPL
- G3 Open-world continual：OFCL
- G4 Ours-Base：仅开集对偶原型元训练（去掉不确定性课程）
- G5 Ours-Full：开集对偶原型 + 不确定性课程（完整方法）

## 2. 公平设置

1. 统一 backbone（ResNet18）
2. 统一数据协议（80+20, 5 sessions）
3. 统一 seeds（建议 3 个）
4. 统一评测轮数（test_times=50）

## 3. 核心表格

- 表1 主结果：Known/Unknown/F1/Incremental/All + AA/PD
- 表2 消融：去 uncertainty、去 dual-prototype、去 dynamic label
- 表3 泛化：Librispeech / NSynth / FMC

## 4. 关键结论

1. Ours-Full 在 AA Unknown 与 AA F1 上优于 FSCIL-only。
2. Ours-Full 在 PD Known 上优于 OSR-only。
3. Ours-Full 相比 Ours-Base 的提升证明不确定性策略有效。

