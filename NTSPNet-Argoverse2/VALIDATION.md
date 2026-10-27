# 已执行验证记录

更新日期：2026-09-22。所有实现修改位于 `PNet-Argoverse2/`。
这份记录区分“工程功能通过”和“经过全量训练得到模型质量结果”；目前完成的是前者。

## 环境

实际使用 `/home/lhb/anaconda3/envs/PNet_av2/bin/python`：
Python 3.8.20、PyTorch 2.4.1、PyG 2.3.1、Lightning 2.0.3、torchmetrics 1.4.0.post0、
pandas 1.4.4、NumPy 1.23.0、PyArrow 17.0.0、Shapely 2.0.6、pytest 7.4.4。
GPU 为 NVIDIA RTX 3090。未升级用户环境。

当前 av2 0.3.1 的官方 reader 在 Python 3.8 下存在类型注解导入错误；
以下真实数据验证使用 `--reader parquet`。它直接读取 AV2 的公开 Parquet/JSON schema。
官方 reader 入口已实现，但尚未在兼容的 Python >=3.9 环境中完成端到端验证。

## 自动测试

```bash
cd /home/lhb/HPNet/PNet-Argoverse2
/home/lhb/anaconda3/envs/PNet_av2/bin/python -m pytest tests -q
```

结果：**20 passed, 8 warnings in 3.30s**。
Warnings 来自既有依赖的弃用提示、PyTorch sparse CSR 提示及 Lightning checkpoint 的 torch.load 提示。

覆盖内容：

- 字符串 ID 和前导零、乱序行、缺帧 mask、未来才出现的目标过滤；
- 去除/扰动未来标签不会改变输入节点、历史运动学、地图或预测输出；
- 邻接与保守换道关系分离，PyG batch 的 lane 索引正确偏移；
- 缓存 round-trip、源文件变化失效、禁止写入原始数据目录；
- 两种预测协议下 baseline/extended 的完整 50/60 帧前向和反向；
- fusion 的零 gate 输出等于 baseline，冻结分支无梯度，gate 可获得梯度；
- 拒绝 INTERACTION checkpoint、无训练步数 checkpoint、不一致 MapEncoder 和验证集统计；
- FDE 最佳模态的 Brier 口径、共同 world 选择、预测坐标还原；
- 提交五列 schema、模态顺序、概率归一化，以及独立检查器对缺失目标的拒绝。

## 真实 AV2 三阶段训练与 checkpoint

调试架构 D=32、heads=4、attention layers=1、hops=1、K=6，H=50/F=60 未缩短。
使用排序后的第一个真实训练场景和第一个真实验证场景，各阶段执行一个 optimizer step：

| 阶段 | 结果 | checkpoint |
|---|---|---|
| baseline | 训练、验证、best/last 保存成功 | `artifacts/smoke/runs/baseline/version_0/checkpoints/last.ckpt` |
| extended provider | 冻结基线地图；Style/Priority 两项辅助监督均非零 | `artifacts/smoke/runs/extended/version_0/checkpoints/last.ckpt` |
| fusion | 加载上述两分支，训练 adapter/gate | `artifacts/smoke/runs/fusion/version_0/checkpoints/last.ckpt` |

Style 校准单独使用有 seed 的两个训练场景，共 1,775 个可靠历史前缀，未使用验证/test。
provider 的首次训练 Style loss=2.032264、Priority loss=1.354616，均参与总损失。

逐张量对比确认 fusion 训练前后：

- 冻结基线 Backbone 不变；
- 冻结 provider Backbone 不变；
- 共享 MapEncoder 与两份初始化 checkpoint 一致；
- 三项 raw gate 分别更新为约 `Style=+0.0005 / Reachability=-0.0005 / Priority=+0.0005`。

另外从 fusion checkpoint 独立恢复，批量验证两个真实 val 场景成功。
这里的误差只是一个优化步之后的 smoke 数值，不具有预测精度或收敛结论。

## 默认完整架构

对真实训练场景 `0000b0f9-99f9-4a1f-a231-5be9e4c523f7`（30 个历史可见智能体）：

- D=128、8 heads、3 attention layers、K=6；
- 输出分别为 `[30,50,6,60,2] / [30,50,6,60,2] / [30,50,6]`；
- GPU 完整前向、轨迹/Style/Priority loss 及反向均成功，全部已产生梯度有限；
- PyTorch 峰值 allocated GPU memory 为 3,717.43 MiB；
- 不包含数据预处理的该次前向＋反向耗时约 1.46 秒。

这是单场景测量，不是全数据集吞吐或显存上界；高密度场景可能显著更大。

## 权重加载与恢复

先用 `--weights` 从 baseline 权重启动新的两轮计划，第一轮后停止；
再用 `--resume` 恢复全部训练状态执行第二轮：

| 状态 | epoch | global_step | 保存时 lr |
|---|---:|---:|---:|
| 恢复前 | 0 | 1 | 0.00025 |
| 恢复后 | 1 | 2 | 0.0（两轮余弦计划完成） |

恢复后参数确实更新；没有把“加载权重”冒充“恢复优化器”。
日志和 checkpoint 位于 `artifacts/smoke/restore_runs/`。

## 测试预测与独立复核

测试只读取无未来真值的原始观测，不计算 test ADE/FDE，不上传服务器。
用前两个测试场景、batch=2 执行并从原始输入独立检查：

| 协议 | 模型 | 场景 | 目标 | Parquet 行数 | 每目标模态/未来帧 |
|---|---|---:|---:|---:|---|
| single-agent | 上述 fusion | 2 | 2 | 12 | 6 / 60 |
| multi-world | 两个真实 train/val 场景上完成一步训练的 extended | 2 | 6 | 36 | 6 / 60 |

两个导出均通过：目标集合精确匹配、概率归一化、所有坐标有限、每模态未来长度60；
multi-world 的同一场景各目标还使用同样的概率和模态顺序。
两个 manifest 均标记 `complete_split=false`。

```bash
python validate_submission.py \
  --submission artifacts/smoke/test_export/submission.parquet --test-limit 2
python validate_submission.py \
  --submission artifacts/smoke/multiworld_export/submission.parquet \
  --protocol multi-world --test-limit 2
```

单目标 Parquet SHA-256：
`9e7fa9d30fa40f483e84ec7277b7a0d79a574fbacafb236d62a743f3d528d1df`

多 world Parquet SHA-256：
`ab801e3bda78ffbb54d0fcb6f9be726811b12d1f2474d84a8a56b2dc0f37e260`

## 其他验证及范围

- 两个真实 val 场景的 `preprocess.py --workers 2` 成功；缓存与数据加载 worker 路径可用。
- 新工程 38 个 Python 文件均通过语法解析。
- SOURCE_MANIFEST 中原 INTERACTION 的16个核心复制源文件哈希均与开始实现时相同。
- 未修改 `1.md`、旧模型源码、旧 checkpoint、INTERACTION 官方测试产物或 AV2 原始数据。
- 未执行全量预处理、全量训练、全量验证、全量测试导出或官方服务器评测。
- 尚无新环境安装验证、多 GPU 支持或官方 reader 的端到端验证。

机器可读汇总：`artifacts/smoke/validation_summary.json`；各阶段日志、环境信息及独立检查 JSON
也在 `artifacts/smoke/`。该目录和缓存均被本工程 `.gitignore` 排除。

## 补充：INTERACTION → AV2 零样本验证（2026-09-22，2026-09-24复核）

已新增独立入口`evaluate_interaction.py`，直接严格加载原V48；协议H10/F30，保持原模型。
固定seed2048随机256个val场景，其中252场景、773目标合格，4场景明确跳过。
联合ADE/FDE/MR：0.873328 m / 2.065985 m / 0.587302。
全部1,339项模型状态验证前后一致，源代码和checkpoint未改动；9月22日27项测试通过。
9月24日独立复核逐场景指标汇总、分母、场景清单、代码和权重哈希，均通过。
这些是自定义跨数据集子集结果，与上面的AV2训练模型smoke验证分开记录。
协议、限制、匀速参考和完整val命令见[ZERO_SHOT.md](ZERO_SHOT.md)。
产物：`artifacts/zero_shot_v48_av2_256_seed2048/`，尚未执行全量跨数据集验证。

## 补充：同目标边际指标与多车子集（2026-09-26复核）

原256场景清单重跑完成，252场景/773目标合格。170个多车场景包含691个目标，
联合ADE/FDE为0.723323/1.706697m，边际ADE/FDE为0.696666/1.609938m；
82个单车场景联合与边际指标完全一致。新增逐车辆明细、场景等权/车辆等权统计和子集分母。
9月24日32项测试通过，9月26日独立CSV重算及模型/适配器/权重哈希复核通过。
旧报告与重跑逐场景指标最大差约1.004mm，所有MR判定相同；相同输入重复GPU推理也出现
毫米级波动。本次联合/边际对照使用同一次forward结果，不混入跨运行的波动。
详见[ZERO_SHOT.md](ZERO_SHOT.md)；原报告保留，新产物位于
`artifacts/zero_shot_v48_av2_256_joint_marginal/`。

## 补充：公开真值测试集复现（2026-09-26）

新增`--split test --test-annotations`，使用AV2官方公开Multi-World标注，仅在模型外计分。
固定seed2048随机256个test场景，253场景/917目标合格，3场景跳过。192个多车场景、
61个单车场景；全部联合ADE/FDE/MR为0.798785m / 1.913608m / 0.616601，
场景等权边际ADE/FDE/actorMR为0.770197m / 1.817336m / 0.320105。
多车联合FDE为1.805682m，边际FDE为1.678823m；单车两类指标一致。
39项测试通过；独立核对原始test目标ID、时间窗、标注匹配、CSV汇总及模型权重哈希均通过。
模型状态不变，未改训练或原始数据。运行约219秒，未执行全量测试、官方协议评测或服务器上传。
详见[ZERO_SHOT.md](ZERO_SHOT.md)；产物位于`artifacts/zero_shot_v48_av2_test256_joint_marginal/`。
