# PNet-Argoverse2

独立的 Argoverse 2 Motion Forecasting 训练、验证与测试预测工程。
AV2 训练流程从本工作区 `HPNet-INTERACTION` 的 PNet 复制并适配模型组件，可独立运行。
另有 `evaluate_interaction.py` 直接加载原 INTERACTION 实现和权重，执行零样本跨数据集验证；
该入口需要保留原工程，提供同目标联合/边际指标与单车/多车子集统计；
支持val评估及配合官方公开Multi-World真值的test离线评估，
详见 [ZERO_SHOT.md](ZERO_SHOT.md)。
不修改 INTERACTION/Argoverse 1 源码或原始数据。复制源文件哈希记录在 `SOURCE_MANIFEST.json`。
本工程没有经过全量训练的 AV2 权重；`artifacts/smoke/` 仅保存接口验证用的小模型，不能用作论文结果。

V48零样本实验的[权重、代码归档与复现说明](REPRODUCIBILITY.md)已单独整理。

## 模型与训练路线

| `--trunk` | 结构与用途 | 训练参数 |
|---|---|---|
| `baseline` | HPNet 原始稠密交互主干，Proposal → detach 中点 anchor → Refinement | 全部参数 |
| `extended`（默认） | 原扩展分支，包含 Style、Reachability、Priority/conflict 和动态 V→L→L→V 传播 | 从头训练；或固定 AV2 基线 MapEncoder 后训练语义分支 |
| `fusion` | 冻结 AV2 基线＋冻结 AV2 语义分支，6 个 adapter、3 个零初始化有界 gate 注入两阶段表示 | 仅 adapter/gate |

`fusion` 对应 INTERACTION 最终模型的处理逻辑。标准 H50/F60 模型不能直接加载 V22/V14/V48 权重：
AV2 的时间窗、数据语义和输出头不同。先在 AV2 上训练 baseline，再用其冻结地图编码器训练 extended，
最后组合两份 checkpoint。初始化会检查训练步数、配置与地图权重一致性。
`extended` 也可独立从头训练和预测，无需任何外部 checkpoint。

默认 H=50、F=60、K=6、D=128、8 heads、3 层交互 attention。
输出保持 `[N,50,6,60,2] / [N,50,6,60,2] / [N,50,6]`，验证和测试使用最后历史锚点。
历史锚点辅助损失权重合计 0.3，最后锚点权重 1；回归使用有效未来位置上的 Smooth L1，
Proposal 最佳模态同时监督 Proposal/Refinement，概率头以 detached refinement 表示进行分类。
扩展模型默认另加 Style CE（0.05）、Priority CE＋顺序损失（0.3）；fusion 不计算这两项辅助损失。

## 环境

当前机器可直接使用已有环境，无需升级它：

```bash
cd /home/lhb/HPNet/PNet-Argoverse2
conda activate PNet_av2
python train.py --help
```

如果 shell 没有激活 conda，把下文 `python` 替换为：

```text
/home/lhb/anaconda3/envs/PNet_av2/bin/python
```

默认 `--reader parquet` 通过 pandas/JSON 实现公开 AV2 schema，避开这台机器 Python 3.8 / av2 0.3.1
导入时的 `Type subscription requires python >= 3.9` 错误。不 monkey-patch 原环境。
`--reader av2` 则使用官方 `load_argoverse_scenario_parquet`、`ArgoverseStaticMap.from_json`；
新建 Python 3.10 环境后安装 `requirements-av2.txt` 可启用该选项。
已验证的是现有环境的 portable reader；官方 reader 在当前不兼容环境下会给出明确错误。

新环境安装示例（按设备安装适合的 PyTorch 后再安装其他依赖）：

```bash
conda create -n pnet-av2-new python=3.10
conda activate pnet-av2-new
python -m pip install -r requirements.txt
# 可选官方 reader：
python -m pip install -r requirements-av2.txt
```

模型无需 Lanelet2 或 Argoverse 1 API。训练、验证支持单 CPU/GPU；尚未提供多 GPU 的严格无重复场景验证。
默认 batch=1、梯度累积=8；保留全部历史智能体时，密集场景计算量更大，可调整半径/隐藏维度。

## 数据、缓存和几何定义

默认原始数据根目录为 `/home/lhb/data-Argoverse2`：

```text
train/<scenario_id>/scenario_<scenario_id>.parquet
train/<scenario_id>/log_map_archive_<scenario_id>.json
val/<scenario_id>/...
test/<scenario_id>/...
```

本地目录计数为 train=199,908、val=24,988、test=24,984。数据读取遵循：

- 50 帧历史、60 帧未来；按 `timestep` 写入张量，缺失帧保持独立 mask。
- 保留字符串 track ID，包括前导零；只加入历史中出现过的智能体。
- `object_type` 与 `object_category` 分开保留；未来出现的智能体不进入上下文。
- 训练目标为非 fragment 的动态智能体；静态物体和 fragment 保留为上下文。
- `observed` 严格等于 `timestep < 50`；test 只接受观测，不读测试标注。
- `csv_velocity/csv_heading` 是沿用主干的字段名称，实际来自 AV2 Parquet。
  其中未来部分始终置零。历史位置差分严格因果、不能跨缺帧。
- 坐标减去 focal agent 在 t=49 的位置，仅平移；导出时加回 float64 origin。
- AV2 MF 没有物体真实长宽，输入置 0 并保留 `size_valid=False`；不写入猜测的尺寸。
- 车道 ID 只在场景内唯一，PyG batch 对边和 lane association 分别正确偏移。
- 首选原 JSON 的 `centerline`，缺少时才用等弧长采样的左右边界中点；左右边界保留原几何。
  两个 reader 共享这一策略，避免 API 插值改变基线输入口径。
- 左右邻居单独记录为 adjacency；只有 `DASHED_WHITE` 邻边进入保守换道候选，
  之后再经过原拓扑构建器的方向与关键区域过滤。这是工程近似，不是法律通行许可。
- AV2 不提供 Lanelet2 `conflicting`。本工程用车道面重叠面积 >0.25m²、方向差和
  高度差过滤构造对称几何冲突候选，排除直接连接/邻接车道。它不代表真实路权。
  Priority 保留原模型“可达集合共享冲突车道”的实例定义；交叉但不共享可达 lane ID 的场景
  可能只通过普通交互图传播，因此该模块并不是完整的几何冲突区域推理器。
- Reachability 的运动预算改为 6 秒，距离上限相应为 160m，最多一次换道。
- 地图限速/交通标志/路权没有可靠输入时保持零占位。当前编码车道边界和中心线；
  原始 drivable-area/crosswalk JSON 可读取，但尚未作为额外网络节点。

缓存默认位于工程内 `cache/av2_v1_<hash>/<split>/`，按 root、schema、reader 和关联配置隔离；
原始文件尺寸/mtime 改变会使对应场景缓存失效。只保存预处理数据，模型 forward 产生的属性不回写缓存。
原始数据目录保持只读。

可以直接训练时按需处理，也可以提前生成缓存：

```bash
python preprocess.py --split train --workers 4
python preprocess.py --split val --workers 4
```

`--manifest`（预处理/统计）或 `--train-manifest/--val-manifest/--test-manifest`（模型入口）
接受每行一个 scenario ID 的文本文件。`--*-limit` 仅用于调试，选取排序后的前 N 个场景。
风格统计默认对训练集进行有种子的随机抽样，不使用验证集。

## 从头训练扩展模型

先生成训练历史的风格伪标签统计；默认随机选择 5,000 个训练场景，并最多均匀保留 200,000 个可靠历史前缀。
这是一份训练子集校准，不宣称是全训练集统计；`--limit 0` 可遍历所选训练集。
统计文件保存场景 ID、seed、schema 和配置，并嵌入后续 checkpoint。

```bash
python compute_style_statistics.py --num-workers 4 \
  --output artifacts/train_style_statistics.pt

python train.py --trunk extended --max-epochs 64 \
  --statistics artifacts/train_style_statistics.pt \
  --output-dir runs/extended_scratch
```

若只做不带 Style 分类监督的调试，可指定 `--lambda-style 0` 跳过统计。
Style 表示和 FiLM 仍通过轨迹损失训练；Priority 辅助损失仍默认启用。

## 仿照最终 INTERACTION 的三阶段训练

三个阶段须使用相同的架构、半径和 protocol。以下均为默认架构；epoch 数只是可运行的配置示例，
不是经过 AV2 收敛实验选出的最佳方案。

```bash
# 1. AV2 基线
python train.py --trunk baseline --max-epochs 64 --output-dir runs/base

# 2. 在相同冻结 MapEncoder 上训练语义分支
python train.py --trunk extended --freeze-map \
  --base-checkpoint /path/to/baseline_best.ckpt \
  --statistics artifacts/train_style_statistics.pt \
  --max-epochs 64 --output-dir runs/provider

# 3. 两个分支冻结，只训练有界残差融合
python train.py --trunk fusion \
  --base-checkpoint /path/to/baseline_best.ckpt \
  --provider-checkpoint /path/to/provider_best.ckpt \
  --lr 1e-3 --warmup-epochs 0 --max-epochs 10 --output-dir runs/fusion
```

每次训练会打印 `BEST_CHECKPOINT` 和 `LAST_CHECKPOINT`。最低 `val_score`（对应 FDE）保留为 best，
同时保存 last。fusion 初始输出与 baseline 相同；D=128 时只训练 100,611 个参数。
概率头与基础分支均冻结，fusion 的分类损失不会直接校准 adapter/gate，置信度改善不能预设。

## 继续训练与独立验证

```bash
# 恢复模型、优化器、调度器、epoch/global_step；架构来自 checkpoint
python train.py --resume /path/to/last.ckpt --max-epochs 64 --output-dir runs/resumed

# 仅加载权重，重新开始优化器；不会加载 INTERACTION/AV1 的不兼容 checkpoint
python train.py --weights /path/to/model.ckpt --lr 1e-4 --max-epochs 10 --output-dir runs/finetune

python val.py --checkpoint /path/to/best.ckpt --output artifacts/validation.json
```

`--resume` 保持 checkpoint 原有学习率计划，应用于中断后的继续训练；若已完成整个计划并需要追加微调，
使用 `--weights` 启动新的学习率计划。传入的 `--max-epochs` 是 Trainer 的停止上限。
风格统计保存在 checkpoint buffer，验证/测试/恢复不需要原统计文件路径。
验证只读取 `val`，目标未来缺帧时明确报错，不把不完整终点当作完整 6 秒 ground truth。

默认 `--protocol single-agent` 只评估 focal target；先以 FDE 选最佳模态，再报告该模态的
`val_minADE`、`val_minFDE`、`val_MR`（FDE>2m）、`val_brier_minFDE=FDE+(1-p)²`，按场景平均。
`--protocol multi-world` 训练时选择场景共享模态；验证 scored/focal actors（category>=2），
以演员平均 FDE 选择共同 world，然后按场景平均输出：
`val_avgMinADE`、`val_avgMinFDE`、`val_actorMR_at_best_world`、`val_brier_avgMinFDE`。
MR 名称刻意明确为“最佳 world 下 miss actor 的比例”，不能直接冒充完整官方 leaderboard 评估套件。
共享 world 概率定义为参与评分智能体的概率几何平均后归一化，验证和测试使用相同公式。
协议写入 checkpoint，验证/测试不会默默切换协议。

## 测试预测与独立完整性检查

```bash
python test.py --checkpoint /path/to/best.ckpt \
  --output-dir artifacts/test_export_v1

python validate_submission.py \
  --submission artifacts/test_export_v1/submission.parquet \
  --protocol single-agent
```

输出采用 AV2 `ChallengeSubmission` 的五列 Parquet 格式：
`scenario_id, track_id, probability, predicted_trajectory_x, predicted_trajectory_y`。
每个目标每个模态一行，x/y 各有 60 个未来坐标；每目标 6 行，而不是 INTERACTION 的逐时间步 CSV。
多 world 情形使用同一套场景概率和模态顺序。文件按 batch 流式写出，不把整个测试预测集合留在内存。

输出目录必须是新的；异常中断保留 `.partial`，成功才改名为 `submission.parquet`。
`manifest.json` 包含 checkpoint/预测 SHA-256、scene/target/row 数、协议、是否覆盖完整测试划分。
独立检查器从原始测试输入重新提取应预测目标，检查目标集合、6 模态、60 帧、有限值及概率。
只导出子集时，导出与检查器必须使用相同 `--test-limit` 或 `--test-manifest`。
原始 test 无未来真值，不输出本地 test ADE/FDE，也不自动上传任何服务器。

## 测试与验证记录

```bash
python -m pytest tests -q
```

最新实际执行范围与限制见 `VALIDATION.md`。

## 官方资料与来源

- [AV2 API 仓库](https://github.com/argoverse/av2-api)
- [Motion Forecasting 数据 schema](https://argoverse.github.io/user-guide/tasks/motion_forecasting.html)
- [地图 API](https://github.com/argoverse/av2-api/blob/main/src/av2/map/map_api.py)
- [场景读取](https://github.com/argoverse/av2-api/blob/main/src/av2/datasets/motion_forecasting/scenario_serialization.py)
- [预测提交格式](https://github.com/argoverse/av2-api/blob/main/src/av2/datasets/motion_forecasting/eval/submission.py)

源码复制部分保留本仓库 Apache-2.0 `LICENSE`；本目录没有复制 AV2 数据或 INTERACTION checkpoint。
