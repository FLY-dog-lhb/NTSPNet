# INTERACTION → AV2 零样本泛化验证

`evaluate_interaction.py` 使用 **INTERACTION 原实现和原 checkpoint**，不训练、不微调、
不重置解码头、不重新估计 style 统计量。与本工程 H50/F60 的 AV2 训练模型分开运行。
目前已验证当前最终 V48（`original_residual`，CSV motion，H10/F30/K6）。

## 运行

使用现有环境，先运行固定种子的随机子集：

```bash
cd /home/lhb/HPNet/PNet-Argoverse2
/home/lhb/anaconda3/envs/PNet_av2/bin/python evaluate_interaction.py \
  --checkpoint ../HPNet-INTERACTION/lightning_logs/version_48/checkpoints/epoch=0-step=2974.ckpt \
  --root /home/lhb/data-Argoverse2 \
  --limit 256 --seed 2048 \
  --output-dir artifacts/my_zero_shot_v48_256
```

完整 validation 命令（移除 `--limit`）：

```bash
/home/lhb/anaconda3/envs/PNet_av2/bin/python evaluate_interaction.py \
  --checkpoint ../HPNet-INTERACTION/lightning_logs/version_48/checkpoints/epoch=0-step=2974.ckpt \
  --root /home/lhb/data-Argoverse2 \
  --output-dir artifacts/zero_shot_v48_full_val
```

默认 GPU、逐场景推理、4 个 CPU 线程；CPU 使用 `--device cpu`。输出目录必须是新目录。
`--manifest path/to/scenario_ids.txt` 可对其他兼容 checkpoint 复用完全相同的场景集合，
不能同时指定 `--limit`。`--interaction-source` 可指定原项目路径。不要在已经导入 AV2
模型模块的 Python 进程中导入原模型；入口会检测同名模块冲突并拒绝混用。

输出包括：`scenario_ids.txt`、逐场景 `per_scene.csv`、运行状态 `run.json` 和完成后的
`summary.json`。失败不会写出完整汇总；已完成行仍保留。汇总记录 checkpoint SHA-256、
源模型代码及适配器哈希、严格加载结果、模型状态是否保持不变、筛选数量和实验协议。

## 固定协议

| 项目 | 定义 |
|---|---|
| 输入 | AV2 timestep 40～49，共10帧；保持10 Hz，名义1秒历史 |
| 真值 | timestep 50～79，共30帧，未来3秒 |
| 模型 | 原模型 forward，输出 `[N,10,K,30,2]`，取最后历史锚点 |
| 目标 | `object_type=vehicle` 且 category 2/3，历史10帧与未来30帧均完整 |
| 上下文 | 在40～49出现的 vehicle、bus、pedestrian、cyclist、motorcyclist |
| 类型映射 | vehicle/bus → car；其他上述动态类别 → pedestrian/bicycle |
| 车辆尺寸 | 长宽均0，表示输入缺失；没有用 AV2 验证集拟合尺寸 |
| 归一化 | 仅平移至 focal 在49帧的位置；不旋转，不改速度单位 |
| 运动信息 | 观测速度与车身heading；行人/两轮车heading按原模型惯例由速度得到 |
| 地图 | AV2 原生中心线；中间点索引与局部切线作为lane参考位置/朝向 |

原 INTERACTION validation 要求完整40帧轨迹，本入口保留完整窗口要求。
不完整轨迹仍能作为历史上下文，但不计分。无合格目标的场景会明确记录并排除分母。
只在标签侧读取未来位置及完整性：传入模型的未来位置、速度、heading、可见性均为0。
AV2 更早的0～39帧也不参与特征、差分、节点选择或 lane association。
原 checkpoint 的 style 统计缓冲区被严格恢复，不读取旧训练机器上的外部统计文件。

AV2 地图无法提供等价的 Lanelet2 法定路权和冲突图：沿用本工程的几何冲突候选，
虚线白线仅作为保守换道边近似，未知限速/路权填0。原模型内部3秒可达性预算等逻辑保持原样，
不使用本工程 AV2 模型修改过的6秒预算或缺帧加速度实现。

## 指标

保留原 INTERACTION 联合指标定义，场景等权：

- `minJointADE`：每个共享模态先平均车辆及30帧误差，再取最佳模态。
- `minJointFDE`：每个共享模态先平均车辆终点误差，再取最佳模态。
- `minJointMR`：每个共享模态取最差车辆的终点误差，再取最优模态；超过2m记为miss。

ADE、FDE、MR各自选择最佳共享模态。不能逐车辆选取不同模态后再称为联合指标。
`focal_*` 额外只统计符合上述筛选的 focal vehicle；其 ADE、FDE同样分别取最小值。
`cv_*` 是相同目标、相同观测下的单模态匀速参考，无拟合过程。它与六模态oracle预测的
模态数量不同，作用是帮助判断量级，不能据此宣称公平的同容量方法比较。

这是 **自定义1秒历史→3秒预测的跨数据集实验**。它不是 AV2 官方5秒→6秒榜单结果，
也不能与原 INTERACTION validation 数值直接作同分布性能比较。测到的变化同时包含
道路/运动分布差异、车辆尺寸缺失、类别映射与地图规则近似的影响。
当前只接受H10/F30、CSV motion、10Hz checkpoint；其他历史长度或输入方案需单独审计。

## 已执行的 V48 零样本验证（2026-09-22，2026-09-24独立复核）

随机种子2048，从24,988个val场景中无放回抽取256个场景；252个场景的773辆车合格，
4个场景无符合协议的目标而跳过。focal车辆合格场景数为226。
在RTX3090上读取、适配、推理和计分耗时约203秒，不包含启动加载开销。

| 指标 | V48，6个共享模态 | 匀速参考，单模态 |
|---|---:|---:|
| minJointADE (m) | 0.873328 | 1.143110 |
| minJointFDE (m) | 2.065985 | 2.771165 |
| minJointMR | 0.587302 | 0.738095 |

focal vehicle的独立oracle minADE / minFDE / MR：1.176843 m / 2.802703 m / 0.438053。
这组初步结果显示V48能在适配后的AV2输入上输出有限预测；不能据此宣称已获得官方AV2性能，
也不能把与INTERACTION原验证分数的差额全部归因于模型泛化。完整val尚未运行。

产物目录：`artifacts/zero_shot_v48_av2_256_seed2048/`。`summary.json`保存完整配置；
`per_scene.csv`保存全部256个场景及跳过状态；`independent_audit.json`记录独立复核。
所有1,339项checkpoint状态严格恢复且评估前后完全一致；33个INTERACTION源文件、
适配器及原checkpoint文件哈希与运行记录一致。V48 checkpoint SHA-256：
`021a3dd8b67ef180cb2b5fcb04d0bd2333322344b1860fc2710b3dc7d64f5eb8`。

2026-09-22本工程测试合计27项通过（8项已有依赖警告）。新增7项覆盖完整输入的未来/早期
历史隔离、完整窗口筛选、历史节点选择、地图参考位置、共享模态oracle和MR定义、
混合模块拒绝、与原INTERACTION指标实现逐项对照。2026-09-24从逐场景CSV独立重算
所有汇总指标并核对场景清单、分母及代码/权重哈希，全部相符。

## 联合与边际对照、多车子集（报告schema v2）

`evaluate_interaction.py`现在默认在一次forward的同一份预测上计算两类指标：

- `minJointADE/FDE`：先对同场景的目标车平均，再选择一个共享最佳模态。
- `marginal_minADE/FDE`：每辆车分别选择自己的最佳模态，再在场景内平均。
- `marginal_actorMR`：每辆车最小FDE是否严格大于2m，再在场景内平均。
- `joint_minus_marginal_ADE/FDE`：上述两种口径的误差之差。

所有上述主表数值先得到逐场景值，再按场景等权汇总。边际ADE与FDE各自独立选模态，
与当前联合诊断保持一致，不声明与官方AV2边际评测的模态选择规则完全相同。
边际结果使用全部合格目标，与只测focal的`focal_*`不同。
`marginal_actor_weighted`另行提供跨全部车辆等权的结果，不用于主表中的联合/边际差值。

`summary.json`增加`subsets.single_target`和`subsets.multi_target`：前者恰有1辆合格目标车，
后者至少有2辆合格目标车。每个子集均记录场景数、目标车数、focal分母、场景等权指标和
车辆等权边际指标。划分依据是参与计分的目标数，不是包含未计分上下文的总节点数。
多车子集仅按数量筛选，不自动等同于具有强交互或冲突的场景。
没有合格目标的场景不进入任何指标分母；空子集指标为null，不伪装成0。
`per_scene.csv`增加子集标签和边际/差值列，`per_actor.csv`保存合格车辆原始字符串ID及边际指标。

在完全相同的目标和预测下，边际ADE/FDE理论上不高于联合ADE/FDE。二者差值反映
**要求车辆共享模态编号带来的oracle误差增量**，不等同于碰撞率，也不是某个交互模块的
因果贡献。单目标场景的联合与边际ADE/FDE应相同。
联合MR是场景级失败事件，边际actorMR是失败车辆比例，二者不可相减后解释成模态一致性损失。

复用原实验清单的命令如下（输出目录必须新建）：

```bash
cd /home/lhb/HPNet/PNet-Argoverse2
/home/lhb/anaconda3/envs/PNet_av2/bin/python evaluate_interaction.py \
  --checkpoint ../HPNet-INTERACTION/lightning_logs/version_48/checkpoints/epoch=0-step=2974.ckpt \
  --manifest artifacts/zero_shot_v48_av2_256_seed2048/scenario_ids.txt \
  --seed 2048 \
  --output-dir artifacts/my_zero_shot_v48_joint_marginal
```

原始H10/F30输入协议、车辆筛选、模型权重及地图适配均保持原样；报告schema升级不改变输入schema。

## 联合/边际及多车子集实测结果（2026-09-24运行，2026-09-26复核）

使用原256场景清单及V48 checkpoint，252场景的773辆车计分，4场景无合格目标。
其中多车子集170场景、691辆车；单车子集82场景、82辆车。以下主表均为场景等权，单位m。

| 子集 | 场景数 | 目标车数 | 联合ADE | 边际ADE | 联合FDE | 边际FDE | FDE差值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全部有效场景 | 252 | 773 | 0.873327 | 0.855344 | 2.065984 | 2.000709 | 0.065274 |
| 多车（至少2辆目标） | 170 | 691 | 0.723323 | 0.696666 | 1.706697 | 1.609938 | 0.096760 |
| 单车（1辆目标） | 82 | 82 | 1.184312 | 1.184312 | 2.810846 | 2.810846 | 0.000000 |

多车子集共享模态约束带来的平均oracle误差增量为ADE 0.026657m、FDE 0.096760m。
单车子集两类指标完全一致。多车与单车子集的交通场景组成不同，不能仅凭多车FDE更低
就断言多车预测更容易；也不能把上述差值归因于某个交互模块。

| 子集 | 联合场景MR | 边际actorMR（先场景内平均，再场景等权） |
|---|---:|---:|
| 全部有效场景 | 0.587302 | 0.315936 |
| 多车 | 0.652941 | 0.250682 |
| 单车 | 0.451220 | 0.451220 |

这两种MR的统计对象不同，不能把差额当作共享模态约束的纯粹影响。
车辆等权边际指标另存于summary的`marginal_actor_weighted`，不与上述场景等权指标混用。

2026-09-24共32项测试通过（8项依赖警告）。2026-09-26独立从773条逐车辆记录重算
场景内均值、全部/单车/多车汇总及车辆等权结果，核对所有分母、目标数与清单，均相符。
同时验证单车退化关系、边际ADE/FDE不高于联合结果，并确认模型、输入适配器和权重哈希。

原报告与本次重跑未达到逐位相等：逐场景旧指标最大绝对差为0.001003742m；
整体联合ADE/FDE差分别为0.000000467m / 0.000001076m；所有原MR判定一致。
对最大偏差场景`bf4d7059-fe77-4cb8-b6da-c29be3abbd59`，使用同一模型及相同图进行3次GPU推理，
也观察到预测坐标最大差0.002595901m。这支持推理数值非确定性的解释；未定位至具体底层算子。
本次联合与边际指标来自同一次forward，因此它们的配对差值不混入不同推理运行的波动。

产物：`artifacts/zero_shot_v48_av2_256_joint_marginal/`，包含：

- `summary.json`：全部及各子集指标、完整实验配置和哈希。
- `per_scene.csv` / `per_actor.csv`：逐场景及逐车辆明细。
- `independent_audit.json` / `audit_report.py`：独立复核结论和可重执行脚本。
- `repeated_inference_check.json`：同一输入重复推理的实测记录。

该实验仍为车辆子集上的自定义1秒历史→3秒预测，未执行全量val；不声明官方AV2成绩。

## 测试集复现实验（公开真值，报告schema v3）

原始`test/`中的parquet只有0～49帧观测。AV2官网现已公开测试真值：
[官方发布及下载页面](https://www.argoverse.org/av2.html)、
[Multi-World测试标注](https://s3.amazonaws.com/argoverse/datasets/av2/motion-forecasting/av2_mf_multi_test_annotations.parquet)。
本工程将标注单独保存在`artifacts/annotations/`，不向原始test文件写入未来数据。
`download_manifest.json`记录来源、下载日期、字节数和SHA-256。

`evaluate_interaction.py --split test`要求显式提供`--test-annotations`。
每个测试场景只从原始parquet构造历史输入、类型、类别及上下文；独立标注模块按
`scenario_id`和原始字符串`track_id`匹配真值，并检查所有scored/focal目标及focal标记一致。
标注每条轨迹有60个未来位置，本实验取前30个，对应timestep 50～79。
外部真值通过`future_frame`参数仅交给标签构造阶段，不参与模型图、lane association、
运动特征或统计量计算。单智能体标注不能替代Multi-World标注。

```bash
cd /home/lhb/HPNet/PNet-Argoverse2
/home/lhb/anaconda3/envs/PNet_av2/bin/python evaluate_interaction.py \
  --checkpoint ../HPNet-INTERACTION/lightning_logs/version_48/checkpoints/epoch=0-step=2974.ckpt \
  --split test \
  --test-annotations artifacts/annotations/av2_mf_multi_test_annotations.parquet \
  --limit 256 --seed 2048 \
  --output-dir artifacts/my_zero_shot_v48_test256
```

省略`--limit`可评价完整test split；目前实际执行范围以结果记录为准。
默认`--split val`保持原用法，且拒绝与`--test-annotations`混用。
新报告增加`complete_split`和测试真值的路径/哈希/时间索引；保留旧`complete_val_split`
字段用于兼容，测试实验中它固定为false。

本实验使用已公开的测试真值进行离线计分，不是官方服务器盲测提交。
仍保持1秒历史→3秒预测、仅车辆目标的自定义协议。随机种子与验证实验相同，但测试和
验证抽取的是不同场景，不能进行逐场景配对比较；不利用测试结果重新选择权重或调整参数。

## 测试集256场景实测结果（2026-09-26）

采用与validation实验相同的V48权重、seed2048和1秒历史→3秒预测协议，
在24,984个测试场景中无放回随机选择256个：253场景、917辆车参与计分，3场景无合格目标。
192个多车场景包含856个目标；61个单车场景包含61个目标。
抽取的测试场景与之前的验证场景不重叠。读取、适配、推理及计分约219秒，未计启动开销。

| 子集 | 场景数 | 目标车数 | 联合ADE (m) | 边际ADE (m) | 联合FDE (m) | 边际FDE (m) |
|---|---:|---:|---:|---:|---:|---:|
| 全部有效场景 | 253 | 917 | 0.798785 | 0.770197 | 1.913608 | 1.817336 |
| 多车 | 192 | 856 | 0.747992 | 0.710322 | 1.805682 | 1.678823 |
| 单车 | 61 | 61 | 0.958657 | 0.958657 | 2.253309 | 2.253309 |

上表均按场景等权，联合与边际使用同一次forward及同一批目标。多车子集FDE差值为
0.126858m，ADE差值为0.037670m；单车两类指标完全一致。

| 子集 | 联合场景MR | 边际actorMR（先场景内平均，再场景等权） |
|---|---:|---:|
| 全部有效场景 | 0.616601 | 0.320105 |
| 多车 | 0.677083 | 0.286389 |
| 单车 | 0.426230 | 0.426230 |

联合MR统计场景级失败，边际actorMR统计车辆失败比例，不能直接把二者相减作为纯粹模态一致性差。
测试与验证子集的场景及目标数量分布不同；两者汇总差异不构成逐场景配对的性能提升/退化证据。

测试真值来源：[AV2官方发布页面](https://www.argoverse.org/av2.html)的
[Multi-World测试标注](https://s3.amazonaws.com/argoverse/datasets/av2/motion-forecasting/av2_mf_multi_test_annotations.parquet)。
完整文件96,109,723字节，97,799条轨迹，覆盖24,984个test场景；本次抽样读取1,014条标注。
标注SHA-256：`f7ba490b7691c9dde3f100e230b173e09cc7fde5c887cc78b775655052dacd45`。
只用每条60帧未来轨迹的前30帧计分，所有模型特征仍由原始test历史和地图产生。

39项测试通过（8项已有依赖警告），新增测试覆盖外部真值与val标签等价性、未来信息隔离、
第50帧对齐、目标/focal匹配、长度/非有限值检查及缺少test真值时的明确拒绝。
独立从原始测试历史重算了各场景目标ID及完整历史筛选，并重算917条逐车辆记录的全部统计。
场景归属、随机种子、测试/验证样本不重叠、公开真值和模型源码/权重哈希均通过复核。
全部1,339项模型状态评估前后一致，没有训练、微调或根据测试结果调整参数。

产物目录：`artifacts/zero_shot_v48_av2_test256_joint_marginal/`：

- `summary.json`：主指标、子集、真值来源哈希与完整配置。
- `per_scene.csv` / `per_actor.csv`：256场景及917辆合格目标的明细。
- `scenario_ids.txt`：本次测试样本清单。
- `independent_audit.json` / `audit_report.py`：独立复核结论与脚本。

本次是使用已公开真值的离线测试实验，属于自定义1秒→3秒车辆子集协议；
不是官方5秒→6秒成绩，也不是未见真值的官方服务器盲测；尚未执行全量test。
