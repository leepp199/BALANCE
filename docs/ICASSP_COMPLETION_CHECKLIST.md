# ICASSP 交付验收清单

该清单是训练完成后的强制交付顺序。未全部勾选前，不将任务标记为完成。

## A. 三数据集重训练与复核

- [ ] FSC-89 不确定性课程版本训练完成
- [ ] LS-100 不确定性课程版本训练完成
- [ ] NS-100 不确定性课程版本训练完成
- [x] 三套当前冻结模型检查点、配置、日志和随机种子归档（见 `experiments/FROZEN_BEST_CONFIGS.md`）
- [ ] 审核基类训练准确率、训练/测试差距和元训练是否真正改善基类表示
- [ ] 核验 LSRB 是否参与基类训练，而非仅离线聚类

## B. 原项目测试形式复现

- [ ] 三个数据集运行原始测试流程
- [ ] 记录 base session、每个 incremental session、平均增量准确率和最终准确率
- [ ] 记录旧类准确率、新类准确率、全类准确率和遗忘量
- [ ] 对异常结果进行数据划分、类别映射和 checkpoint 审核

## C. 混合开放世界主实验

- [ ] 三个数据集运行正常混合已知/未知流
- [ ] 记录 known/unknown recall、F1、AUROC、AUPR、FPR95
- [ ] 记录 CANA 的 ACC、NMI、ARI、候选纯度和新类覆盖率
- [ ] 使用 50 次独立重复，输出均值、标准差和 95% 置信区间（最终验收固定为 50 次，10 次结果仅作筛选）
- [ ] 生成 paired significance test 和 bootstrap interval

## D. SOTA 对比

- [ ] TEEN
- [ ] OFCL
- [ ] OPCR
- [ ] YLOC
- [ ] Happy
- [ ] CaMP
- [ ] MetaGCD
- [ ] OCGCD
- [ ] OpenIncrement
- [ ] VB-CGCD
- [ ] 每个方法注明 exact reproduction、official transfer 或 protocol-compatible reimplementation
- [ ] 不把不兼容协议的结果伪装成同条件 SOTA

## E. 独立消融实验

- [ ] baseline
- [ ] LSRB 离线、不参与训练
- [ ] LSRB 联合训练
- [ ] CANA
- [ ] 不确定性记忆
- [ ] 均匀回放与困难样本回放
- [ ] 可靠性门控
- [ ] 有界残差更新与无界更新
- [ ] 核范数、熵、互信息和预测分散性的单项/组合比较
- [ ] LSRB bank size、结构损失权重、协方差收缩、回放强度和残差步长敏感性

## F. 论文图表

- [ ] LSRB 联合训练与混合开放流总框架图
- [x] 每 session incremental/all accuracy 曲线（LS/NS 使用 50 次正式结果；FSC 仍使用已有重复结果）
- [ ] 遗忘曲线
- [ ] OSR ROC 与 PR 曲线
- [ ] CANA ACC/NMI/ARI 曲线
- [ ] 消融热图或分组柱状图
- [ ] 原型漂移和稳定性图
- [ ] 内存、运行时间和参数量对比图
- [ ] 使用统一 Image2 风格、字体、颜色和图例

## G. Markdown 主稿

- [ ] 主实验表只放复现结果和 SOTA 对比
- [ ] 消融实验单独成表
- [ ] 填入 50 次重复的最终数值
- [ ] 补充实验设置、统计检验和失败案例
- [ ] 全文统一使用 LSRB/CANA，旧 DFSB/BCD 仅在复现说明中出现
- [ ] 逐条回应审稿意见
- [ ] 参考文献与正文引用逐项核对
- [ ] 生成最终投稿版和补充材料

## H. 最终性能验收门槛

- [x] NS-100 在冻结配置的 50 次独立重复上平均 `inc_acc >= 65%`（70.16%）
- [ ] FSC-89 在冻结配置的 50 次独立重复上平均 `inc_acc >= 60%`
- [ ] 两个数据集均保留逐 round、逐 session 原始记录，并报告样本标准差和 95% 置信区间
- [ ] 先用独立验证/筛选种子冻结超参数，再运行 50 次正式测试；不得按测试结果挑选配置
- [x] 完成 layer2、layer3、layer4 及预先选定多层融合的结构特征筛选消融（FSC 单流；layer4 最优）
- [ ] 所有论文图仅使用 LSRB、CANA、UMR 和 FOWAC 正式名称重新生成
