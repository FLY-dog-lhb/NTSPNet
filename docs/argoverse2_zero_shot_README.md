# NTSPNet: INTERACTION-to-Argoverse 2 Zero-Shot Evaluation

This directory documents and implements the zero-shot cross-dataset evaluation of NTSPNet on the Argoverse 2 (AV2) Motion Forecasting dataset. The evaluated checkpoint was trained only on the INTERACTION dataset. No AV2 training, fine-tuning, optimizer update, style-statistics recalibration, or target-domain parameter fitting is performed.

The evaluation uses a custom protocol that preserves the temporal interface of the INTERACTION-trained model:

- observation: 10 frames at 10 Hz (1 s), AV2 timesteps 40--49;
- prediction: 30 frames at 10 Hz (3 s), AV2 timesteps 50--79;
- modes: 6 shared trajectory modes;
- targets: scored or focal vehicles with complete 10-frame histories and complete 30-frame futures;
- coordinate system: translation to the focal vehicle position at timestep 49, without rotation.

This is not the official AV2 5 s observation-to-6 s prediction leaderboard protocol. The reported numbers must therefore be interpreted as a controlled cross-dataset transfer experiment, not as official AV2 leaderboard results.

## 1. Repository layout

The commands below assume the following layout:

```text
NTSPNet/
├── NTSPNet-INTERACTION/
├── NTSPNet-Argoverse2-ZeroShot/
│   ├── README.md
│   ├── evaluate_interaction.py
│   ├── requirements.txt
│   ├── cross_domain/
│   ├── datasets/
│   ├── manifests/
│   │   ├── av2_val_seed2048_256.txt
│   │   └── av2_test_seed2048_256.txt
│   └── results/
│       └── v22_vs_ntspnet/
└── checkpoints/
    ├── ntspnet_final_seed2048.ckpt
    └── hpnet_local_baseline_v22.ckpt
```

The original AV2 dataset is not redistributed in this repository. Obtain it from the official Argoverse 2 provider and accept the corresponding terms of use.

## 2. Released checkpoints

The NTSPNet checkpoint used in the zero-shot experiment is:

```text
File: ntspnet_final_seed2048.ckpt
Release: v1.0.2
SHA-256: 021a3dd8b67ef180cb2b5fcb04d0bd2333322344b1860fc2710b3dc7d64f5eb8
URL: https://github.com/FLY-dog-lhb/NTSPNet/releases/download/v1.0.2/ntspnet_final_seed2048.ckpt
```

It is the final INTERACTION-trained V48 model with `trunk_variant=original_residual`, `H=10`, `F=30`, `K=6`, CSV motion inputs, and seed 2048.

The locally reproduced HPNet baseline used for the paired comparison is:

```text
File: hpnet_local_baseline_v22.ckpt
Release: v1.0.2
SHA-256: 8cafeec97aa7504628db10237ab8ae535e0732f811f3537478a3e56c2aed737e
URL: https://github.com/FLY-dog-lhb/NTSPNet/releases/download/v1.0.2/hpnet_local_baseline_v22.ckpt
```

Download and verify both checkpoints:

```bash
mkdir -p checkpoints

curl -L \
  https://github.com/FLY-dog-lhb/NTSPNet/releases/download/v1.0.2/ntspnet_final_seed2048.ckpt \
  -o checkpoints/ntspnet_final_seed2048.ckpt

curl -L \
  https://github.com/FLY-dog-lhb/NTSPNet/releases/download/v1.0.2/hpnet_local_baseline_v22.ckpt \
  -o checkpoints/hpnet_local_baseline_v22.ckpt

printf '%s  %s\n' \
  021a3dd8b67ef180cb2b5fcb04d0bd2333322344b1860fc2710b3dc7d64f5eb8 \
  checkpoints/ntspnet_final_seed2048.ckpt \
  8cafeec97aa7504628db10237ab8ae535e0732f811f3537478a3e56c2aed737e \
  checkpoints/hpnet_local_baseline_v22.ckpt | sha256sum -c -
```

Neither `semantic_provider_v14.ckpt` nor an external style-statistics file is loaded during this evaluation. The final checkpoint already contains the complete model state and its style-statistics buffers.

## 3. AV2 data layout

The evaluator expects the public AV2 Motion Forecasting directory structure:

```text
/path/to/Argoverse2/
├── val/
│   └── <scenario_id>/
│       ├── scenario_<scenario_id>.parquet
│       └── log_map_archive_<scenario_id>.json
└── test/
    └── <scenario_id>/
        ├── scenario_<scenario_id>.parquet
        └── log_map_archive_<scenario_id>.json
```

Validation future trajectories are read from the validation scenario Parquet files. The original AV2 test files contain histories only. Offline test evaluation therefore additionally requires the publicly released Multi-World test annotations:

```text
av2_mf_multi_test_annotations.parquet
```

The test annotations are passed only to the label construction and metric code. They are never included in the graph or passed to the model.

## 4. AV2-to-NTSPNet input conversion

No offline conversion into a new dataset format is required. Each AV2 scenario is converted to a PyTorch Geometric `HeteroData` graph at evaluation time:

```text
AV2 scenario Parquet + static-map JSON
  -> AV2 table and map parsing
  -> causal 10-frame history extraction
  -> NTSPNet-compatible agent and lane graph
  -> frozen INTERACTION model forward pass
  -> separate future-label construction
  -> metric aggregation
```

### 4.1 Temporal selection and leakage prevention

- Only timesteps 40--49 determine model nodes and features.
- Timesteps 50--79 are used only as evaluation labels.
- Timesteps 0--39 and 80--109 are ignored by this protocol.
- The focal vehicle must be present at timestep 49.
- The graph is constructed before future labels are attached.
- For test evaluation, the external annotation table is passed through a label-only interface.
- Model parameters are compared before and after evaluation; any state change is treated as an error.

### 4.2 Agent conversion

The adapter keeps dynamic context agents observed during timesteps 40--49:

```text
vehicle, bus, pedestrian, cyclist, motorcyclist
```

The INTERACTION-compatible type mapping is:

```text
vehicle or bus                    -> car type
pedestrian/cyclist/motorcyclist  -> pedestrian/bicycle type
```

Scored prediction targets satisfy all of the following:

1. `object_type == vehicle`;
2. `object_category >= 2` (scored or focal);
3. all 10 historical frames are available;
4. all 30 future frames are available.

The focal vehicle position at timestep 49 is used as the local origin. Positions and map geometry are translated by this origin, but the coordinate system is not rotated. AV2 velocities are retained in metres per second. Vehicle and bus headings use the AV2 heading field; headings for pedestrians and two-wheel agents are computed from their velocity directions.

AV2 Motion Forecasting does not provide measured actor dimensions equivalent to the INTERACTION inputs. Consequently, `length` and `width` are both set to zero for all agents. No dimensions or target-domain statistics are estimated from AV2.

The main tensors constructed by the adapter are:

| Tensor | Shape | Description |
|---|---:|---|
| `agent.position` | `[N, 40, 2]` | compatibility container; slots 0--9 store AV2 timesteps 40--49 |
| `agent.visible_mask` | `[N, 40]` | historical availability mask |
| `agent.csv_velocity` | `[N, 40, 2]` | observed AV2 velocity |
| `agent.csv_heading` | `[N, 40]` | observed or velocity-derived heading |
| evaluation target | `[N, 30, 2]` | AV2 timesteps 50--79, kept outside the model graph |
| refined prediction | `[N, 6, 30, 2]` | final-history-anchor, six-mode prediction |

The complete NTSPNet forward output has shape `[N, 10, 6, 30, 2]`. Evaluation selects the refined prediction at the final historical anchor.

## 5. AV2 map adaptation

Each `log_map_archive_<scenario_id>.json` is converted into the lane, polyline, and lane-to-lane relations expected by the INTERACTION-trained model.

### 5.1 Lane geometry

- The native AV2 centerline is used when available.
- If a centerline is unavailable, the left and right boundaries are independently resampled to 20 points and averaged.
- The lane reference position is the middle-index centerline point.
- The lane heading is the local tangent at that reference point.
- Lane length, mean width, curvature, turn category, endpoint positions, and centerline spacing are computed from AV2 geometry.
- Left boundary, centerline, and right boundary segments are represented as polyline nodes with the source side encoding `0/1/2`.

### 5.2 Lane topology and lane changes

The adapter preserves AV2 predecessor, successor, left/right adjacency, lane type, and intersection attributes. A neighboring lane is treated as a conservative lane-change candidate only when the shared AV2 boundary is `DASHED_WHITE`. Solid, mixed, yellow, and unknown boundaries do not create lane-change edges.

This is an engineering approximation for compatibility and must not be interpreted as a legal right-of-way rule.

### 5.3 Geometric conflict approximation

AV2 does not provide the Lanelet2 conflict graph used by the INTERACTION pipeline. The adapter therefore creates a symmetric geometric conflict candidate when all of the following hold:

1. both lanes are drivable `VEHICLE` or `BUS` lanes;
2. the lanes are not already connected or adjacent;
3. their median height difference is at most 2 m;
4. they are not approximately parallel (15-degree filtering threshold);
5. their lane polygons overlap by more than 0.25 square metres.

The resulting relations are passed through the same potential-conflict topology and lane-association/reachability interfaces used by NTSPNet. Fields without an equivalent AV2 source are not inferred:

```text
speed_limit = 0
has_traffic_sign = 0
legal right-of-way features = 0
```

These approximations are fixed and identical for the baseline and NTSPNet comparison.

## 6. Environment

Create an environment compatible with the versions listed in `requirements.txt`. A typical setup is:

```bash
conda create -n ntspnet-av2 python=3.10
conda activate ntspnet-av2

# Install a PyTorch build matching the local CUDA runtime first.
python -m pip install -r requirements.txt
```

The zero-shot entry point loads `NTSPNet-INTERACTION/model/PNet.py` directly. It does not use an independently trained AV2 model or an AV2-specific checkpoint.

## 7. Evaluation commands

Run every checkpoint in a fresh Python process. The checkpoint loader imports the INTERACTION implementation first and rejects conflicting model/module imports.

### 7.1 Validation subset used in the recorded comparison

From `NTSPNet-Argoverse2-ZeroShot/`:

```bash
python evaluate_interaction.py \
  --interaction-source ../NTSPNet-INTERACTION \
  --checkpoint ../checkpoints/ntspnet_final_seed2048.ckpt \
  --root /path/to/Argoverse2 \
  --split val \
  --manifest manifests/av2_val_seed2048_256.txt \
  --seed 2048 \
  --device cuda \
  --threads 4 \
  --output-dir results/ntspnet_val_seed2048_256
```

Run the baseline on exactly the same scenarios:

```bash
python evaluate_interaction.py \
  --interaction-source ../NTSPNet-INTERACTION \
  --checkpoint ../checkpoints/hpnet_local_baseline_v22.ckpt \
  --root /path/to/Argoverse2 \
  --split val \
  --manifest manifests/av2_val_seed2048_256.txt \
  --seed 2048 \
  --device cuda \
  --threads 4 \
  --output-dir results/baseline_val_seed2048_256
```

To evaluate the complete validation split, omit `--manifest`. The recorded experiment described below used the fixed 256-scenario manifest; it was not a complete-split evaluation.

### 7.2 Offline test-subset evaluation

```bash
python evaluate_interaction.py \
  --interaction-source ../NTSPNet-INTERACTION \
  --checkpoint ../checkpoints/ntspnet_final_seed2048.ckpt \
  --root /path/to/Argoverse2 \
  --split test \
  --test-annotations /path/to/av2_mf_multi_test_annotations.parquet \
  --manifest manifests/av2_test_seed2048_256.txt \
  --seed 2048 \
  --device cuda \
  --threads 4 \
  --output-dir results/ntspnet_test_seed2048_256
```

For the paired baseline evaluation, replace the checkpoint with `hpnet_local_baseline_v22.ckpt` and select a new output directory. Output directories must not already exist, which prevents accidental overwriting of previous evidence.

## 8. Evaluation outputs

Each run produces:

```text
scenario_ids.txt  selected scenario IDs
run.json          protocol, code hashes, checkpoint provenance, and status
summary.json      final metrics and denominators
per_scene.csv     per-scene joint and marginal metrics
per_actor.csv     per-target marginal metrics
```

The evaluator additionally records:

- checkpoint SHA-256;
- strict checkpoint-loading status;
- equality of all loaded checkpoint tensors;
- source-code and adapter hashes;
- whether model state remained unchanged;
- selected, evaluated, and skipped scene counts;
- evaluated target count;
- the exact temporal and map-adaptation protocol.

## 9. Metrics

For a scene with multiple scored vehicles, the joint metrics require a shared mode index across all targets:

- `minJointADE`: average displacement error over targets and 30 future frames, minimized over the six shared modes;
- `minJointFDE`: average final displacement error over targets, minimized over the six shared modes;
- `minJointMR`: the best shared mode is a miss when its worst target final error is greater than 2 m.

ADE and FDE independently select their minimizing shared modes. Scene-level values are averaged with equal scene weight.

The report also provides marginal diagnostics in which each vehicle may select its own best mode. Marginal metrics are supplementary and are not interchangeable with the joint metrics.

## 10. Recorded paired-subset results

The archived comparison used the same fixed scene manifests, target sets, adapter, and metric implementation for both checkpoints.



## 11. Zero-shot integrity guarantees

The implementation enforces the following conditions:

1. `load_state_dict(..., strict=True)` is required;
2. the model is set to `requires_grad_(False)` and `eval()`;
3. no optimizer or training loop is created;
4. no AV2 statistics are fitted or written into the model;
5. checkpoint state is compared before and after inference;
6. future labels cannot enter the model graph;
7. baseline and NTSPNet use identical scenario manifests, targets, map conversion, and metrics;
8. the complete checkpoint and adapter provenance is written to the run report.

These checks establish an auditable zero-shot evaluation procedure. They do not remove the domain gap between INTERACTION and AV2, including differences in actor dimensions, map semantics, right-of-way information, and the official temporal protocol.

## 12. Citation and reproducibility statement

When reporting this experiment, describe it as a custom INTERACTION-to-AV2 zero-shot transfer evaluation. Do not describe it as an official AV2 leaderboard result. A suitable reproducibility statement is:

> The NTSPNet and locally reproduced HPNet checkpoints were trained exclusively on INTERACTION and evaluated on Argoverse 2 without training, fine-tuning, optimizer updates, or target-domain statistics recalibration. Both models used the same deterministic AV2-to-INTERACTION adapter, fixed scenario manifests, target definitions, map approximations, and metric implementation. The custom protocol uses 1 s of history to predict 3 s of future motion and is not the official AV2 leaderboard protocol.

