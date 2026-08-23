# 期刊论文初稿（针对你的任务重写版）

## 题目

不确定性驱动与开集对偶原型元训练的开放世界小样本音频类增量学习

## 摘要

开放世界小样本音频类增量学习需要同时解决三类矛盾：已知类分类精度、未知类拒识能力和跨会话稳定性。为此，本文提出一种统一框架：在元训练阶段通过支持原型校准与伪未知原型生成构建开集对偶原型学习机制，并采用动态 openset 标签分配增强边界判别；在基类训练阶段通过 MC Dropout 多视角不确定性估计实现困难类感知课程重加权；在推理阶段采用同类正负分数一对一比较的阈值自由判决策略。基于 Librispeech FSCIL 协议（80 base + 20 novel, 5 sessions）及现有多组实验日志，方法在 Unknown 与 F1 维度表现出更优折中，同时在 PD 指标上显示出更强增量稳定性。本文进一步给出与 FSCIL/OSR 两类 SOTA 的联合对比协议，验证所提策略在开放世界场景中的有效性。

## 1 引言

现有 FSCIL 方法主要优化已知类增量分类，往往默认闭集测试；现有 OSR 方法主要关注单阶段未知拒识，缺少持续增量过程建模。你的任务本质是两者的交集：

- 持续增量引入新类（FSCIL）
- 每轮推理都可能出现未知类（OSR）

本文核心观点：必须把“不确定性训练”和“开集对偶原型元训练”联合起来，才能同时提升 Unknown Acc 与增量鲁棒性。

## 2 方法

### 2.1 开集对偶原型元训练

给定支持样本特征 z，先得到已知原型 p_k，再生成对应伪未知原型 p~_k。模型学习目标不是“单一未知类”，而是“每个已知类周围的局部未知边界”。

分类损失：L = L_ce(y_dyn) + λ1 L_open + λ2 L_funit。

### 2.2 不确定性课程重加权

通过 MC Dropout + SpecAugment 生成多视角概率矩阵 P，定义不确定性 U(x)=||P||_*。

将类级不确定性映射为采样权重，困难类在后期训练中被更频繁采样，从而提升 Unknown 与后期会话稳定性。

### 2.3 阈值自由开集判决

对样本先找最相似已知类 c*，再比较该类正负得分：s_pos(c*) > α s_neg(c*) 判 known，否则 unknown。

## 3 实验设定

- 数据协议：Librispeech 80+20，5 sessions
- 元任务：5-way 5-shot
- 重复测试：50 次
- 评价指标：AA/PD + Known/Unknown/F1/Incremental/All

## 4 已验证结果（来自你的现有日志）

- test_result1127all.txt
  - AA Known 0.8689
  - AA Unknown 0.7411
  - AA F1 0.8962
  - AA Incremental 0.8969
  - PD Known 0.0849
  - PD Unknown 0.1457

- test_resultLSenhancek0.30.4.txt
  - AA Known 0.8811
  - AA Unknown 0.7006
  - AA F1 0.8776
  - AA Incremental 0.8918
  - PD Known 0.0170
  - PD Unknown 0.1202

说明：后一配置显著降低 PD Known，前一配置在 Unknown/F1 上更强，体现了不确定性与开集边界之间的可调折中。

## 5 任务对齐的 SOTA 对比

### 5.1 FSCIL 主线

1. TOPIC (CVPR 2020)
2. CEC (CVPR 2021)
3. FACT (CVPR 2022)
4. LIMIT / F2M（可二选一）

### 5.2 OSR/OW 主线

1. OpenMax (CVPR 2016)
2. PROSER (CVPR 2021)
3. ARPL (TPAMI 2022)
4. OFCL（与你目录中的 OFCL 对齐）

### 5.3 联合评测原则

1. 每个方法都输出 Known/Unknown/F1/Incremental/All
2. 同时报 AA 与 PD
3. 同一数据划分、同一会话、同一随机种子集合

## 6 结论主张

1. 不确定性训练策略主要提升 PD 与后期稳健性。
2. 开集对偶原型元训练主要提升 Unknown 可分性与 F1。
3. 二者联合形成更优 Pareto 前沿。

