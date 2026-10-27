# 两次AV2零样本实验的模型与权重

validation和test实验都使用 **HPNet-INTERACTION的PNet/V48最终模型**，
`trunk_variant=original_residual`，H10/F30/K6；两次使用同一个完整checkpoint。
没有使用H50/F60的AV2训练模型，也没有用AV2数据微调。

归档权重：[INTERACTION_V48_epoch0_step2974.ckpt](artifacts/reproducibility_v48_av2/weights/INTERACTION_V48_epoch0_step2974.ckpt)。
原始位置：`../HPNet-INTERACTION/lightning_logs/version_48/checkpoints/epoch=0-step=2974.ckpt`。
文件大小50,443,836字节，SHA-256：

```text
021a3dd8b67ef180cb2b5fcb04d0bd2333322344b1860fc2710b3dc7d64f5eb8
```

| 对应实验 | 有效场景/目标 | 联合ADE/FDE (m) | 使用权重 |
|---|---|---|---|
| validation，seed2048抽256场景 | 252 / 773 | 0.873327 / 2.065984 | 同一V48 checkpoint |
| test，seed2048抽256场景 | 253 / 917 | 0.798785 / 1.913608 | 同一V48 checkpoint |

为增强可复现性，已归档原模型33个源文件、各次实验对应版本的适配器、场景清单、
原始报告/CSV、完整权重、公开测试真值及环境版本，并附文件校验和。
验证入口历史版本的哈希与原报告逐字节一致；归档在独立目录内，不回滚工作区代码。
整个归档位于被Git忽略的`artifacts/reproducibility_v48_av2/`，迁移时请单独复制完整目录；
原始AV2场景及地图仍需另行提供。

[归档说明与复现命令](artifacts/reproducibility_v48_av2/README.md) ·
[文件清单](artifacts/reproducibility_v48_av2/MANIFEST.json) ·
[归档加载/推理检查](artifacts/reproducibility_v48_av2/packaging_check.json)

使用已有PNet_av2解释器运行归档中的`reproduce.py --verify-only`可核对完整性；
`--split val`或`--split test`会选择对应历史适配器并使用原场景清单。
GPU推理存在已测得的毫米级波动，不能承诺结果所有小数位逐位一致。
本归档对应自定义1秒历史→3秒预测实验，包含两次256场景子集，非官方完整AV2协议结果。
