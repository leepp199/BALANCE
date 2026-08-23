# baseline 模块说明（面向论文撰写）

## 1. 主流程脚本

- `train_unopenset.py`：当前主实验入口，负责 base 训练、open-set 元训练、增量会话评测与 AA/PD 指标输出。
- `train.py`：历史主流程版本，可作为旧基线对照。
- `train_openset_vaze.py`：开集方向的补充训练入口。

## 2. 网络与方法模块

- `network.py`
  - `MYNET` 主网络：Log-Mel 前端 + ResNet18 编码 + FC 原型头。
  - `mode='openmeta'` 下执行开集元训练逻辑。
  - `task_proto()` 中实现动态 openset 标签分配。
  - `get_uncertainty()` 中实现 MC Dropout 不确定性估计。

- `models/AttnClassifier.py`
  - `SupportCalibrator`：支持集原型校准。
  - `OpenSetGenerater`：伪未知原型生成。
  - `Metric_Cosine`：余弦度量打分。
  - 这是“开集对偶原型元训练”的核心模块。

- `models/metatrainer_oo.py`
  - 开集元训练主循环。
  - 训练时同时统计闭集准确率和 open-set AUROC。

- `models/uncertainty.py`
  - 类级不确定性统计（用于课程学习/重加权）。

## 3. 数据与采样

- `data/dataloader.py`：pretrain/openmeta/test 各阶段 dataloader。
- `data/librispeech.py`、`data/nsynth.py`、`data/FMC.py`：多数据集协议定义。
- `data/sampler.py`：episode 采样策略。

## 4. 开集判决与评测

- `threshold_free.py`
  - 已知/未知判决：同类别正负分 1 对 1 比较（而非全局单阈值）。

- `models/FSEval.py`
  - meta 评测工具：Acc、AUROC、F-score。

## 5. 指标输出（train_unopenset）

- Session 级：Known Acc / Unknown Acc / F1 / Incremental Acc / All Acc
- 汇总级：
  - AA：4 个增量 session 的平均值
  - PD：Session1 - Session4 的性能下降

