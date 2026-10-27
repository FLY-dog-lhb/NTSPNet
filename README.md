# NTSPNet

NTSPNet is a multi-agent trajectory prediction model for the
[INTERACTION dataset](https://interaction-dataset.com/). It extends the HPNet
joint-prediction backbone with driving-style representation, lane reachability,
priority/conflict reasoning, and lane-based traffic propagation while preserving
the original HPNet prediction path.

This repository contains the paper implementation, exact full-data training
recipes, validation code, ablation recipes, and a manifest for all released
checkpoints. Reported numbers below are validation-set results over 11,794
INTERACTION scenarios; they are not official blind-test leaderboard scores.

## Model overview

The released model uses two frozen feature paths and a trainable residual
fusion layer:

1. A frozen local HPNet V22 trunk remains the primary proposal/refinement path.
2. A frozen semantic provider supplies Driving Style, Reachability, Priority,
   conflict-aware interaction, and Vehicle-to-Lane-to-Lane-to-Vehicle context.
3. Six zero-initialized representation adapters inject proposal/refinement
   corrections through three learned module gates.
4. The HPNet proposal and refinement heads produce six joint trajectory modes.

Only the residual fusion parameters are optimized in the released one-epoch
recipe. The final model therefore starts from the exact local V22 prediction
function and learns bounded corrections without replacing the baseline trunk.

For the full method description, see
[`docs/final_model_spec.md`](docs/final_model_spec.md).

## Validation results

All entries use the same 11,794-scene validation set and the same CSV
kinematics/V22 map semantics. "Local baseline V22" is the checkpoint used by
this project; it is not claimed to be numerically identical to the checkpoint
or table entry released with the original HPNet paper.

| Model | Active residual modules | Seed | minJointADE | minJointFDE | minJointMR |
|---|---|---:|---:|---:|---:|
| Local baseline V22 | None | - | 0.17112444 | 0.54093823 | 0.12260471 |
| Style only | Style | 2048 | 0.16715579 | 0.53781700 | 0.11895879 |
| Reachability only | Reachability | 2048 | 0.16926204 | 0.53986531 | 0.12082415 |
| Priority only | Priority | 2048 | 0.16724522 | 0.53742462 | 0.11929795 |
| Style + Reachability | Style, Reachability | 2048 | 0.16685672 | 0.53717697 | 0.11794133 |
| Style + Priority | Style, Priority | 2048 | 0.16599800 | 0.53602624 | 0.11760217 |
| Reachability + Priority | Reachability, Priority | 2048 | 0.16723168 | 0.53737497 | 0.11870442 |
| **NTSPNet** | **All three** | **2048** | **0.16599056** | **0.53593224** | **0.11785654** |
| NTSPNet | All three | 1024 | 0.16575511 | 0.53539568 | 0.11819569 |
| NTSPNet | All three | 4096 | 0.16586624 | 0.53585106 | 0.11751738 |

Exact asset names, file sizes, SHA-256 hashes, and metric values are recorded
in [`NTSPNet-INTERACTION/configs/released_checkpoints.csv`](NTSPNet-INTERACTION/configs/released_checkpoints.csv).

# NTSPNet: INTERACTION-to-Argoverse 2 Zero-Shot Evaluation

This directory documents and implements the zero-shot cross-dataset evaluation of NTSPNet on the Argoverse 2 (AV2) Motion Forecasting dataset. The evaluated checkpoint was trained only on the INTERACTION dataset. No AV2 training, fine-tuning, optimizer update, style-statistics recalibration, or target-domain parameter fitting is performed.

The evaluation uses a custom protocol that preserves the temporal interface of the INTERACTION-trained model:

- observation: 10 frames at 10 Hz (1 s), AV2 timesteps 40--49;
- prediction: 30 frames at 10 Hz (3 s), AV2 timesteps 50--79;
- modes: 6 shared trajectory modes;
- targets: scored or focal vehicles with complete 10-frame histories and complete 30-frame futures;
- coordinate system: translation to the focal vehicle position at timestep 49, without rotation.

This is not the official AV2 5 s observation-to-6 s prediction leaderboard protocol. The reported numbers must therefore be interpreted as a controlled cross-dataset transfer experiment, not as official AV2 leaderboard results.

## Installation

The paper environment used Python 3.8, PyTorch 2.4.1 with CUDA 12.1,
PyTorch Lightning 2.0.3, PyTorch Geometric 2.3.1, and lanelet2.

```bash
conda env create -f NTSPNet-INTERACTION/environment.yml
conda activate ntspnet
```

If lanelet2 is not available through your selected conda channels, install its
Python bindings separately for the same environment. A pip-oriented package
list is also provided in `NTSPNet-INTERACTION/requirements.txt`.

## Dataset preparation

Obtain INTERACTION under its own license from the
[official Hugging Face dataset page](https://huggingface.co/datasets/INTERACTION-dataset/INTERACTION-dataset)
and arrange the official files as shown below. See [`DATA.md`](DATA.md) for
licensing, provenance, derived-artifact roles, integrity checks, and the
one-command paper-data preparation path.

```text
/path/to/INTERACTION/
|-- maps/*.osm
|-- train/*.csv
|-- val/*.csv
`-- test_multi-agent/*.csv       # only needed for challenge submission
```

Set a reusable path. The commands below are run from the repository root:

```bash
export DATA_ROOT=/path/to/INTERACTION
```

For the exact paper setup, download `train_style_statistics.pt` and
`SHA256SUMS` from the v1.0.2 GitHub Release. The 18 exact map-semantic files
are stored individually in this repository. Then run:

```bash
export PAPER_ASSET_DIR=/path/to/downloaded/release/assets
bash NTSPNet-INTERACTION/scripts/prepare_paper_data.sh
```

The equivalent manual steps follow.

Create the PyG processed scenarios:

```bash
python NTSPNet-INTERACTION/prepare_interaction_data.py \
  --root "$DATA_ROOT" --split train val
```

Generate the CSV-motion sidecars used by every reported experiment:

```bash
python NTSPNet-INTERACTION/compute_csv_motion_cache.py \
  --root "$DATA_ROOT" \
  --output_dir "$DATA_ROOT/csv_motion_v1" \
  --split train val
```

For exact paper reproduction, copy the unarchived map files from the
repository and the released Style statistics into the dataset root:

```bash
cp -a NTSPNet-INTERACTION/reproducibility_assets/v22_map_geometry_v1 \
  "$DATA_ROOT/"
cp "$PAPER_ASSET_DIR/train_style_statistics.pt" "$DATA_ROOT/"
```

The resulting paths are:

```text
/path/to/INTERACTION/
|-- train_style_statistics.pt
`-- v22_map_geometry_v1/*.pt
```

The map cache can alternatively be regenerated from the official map files:

```bash
python NTSPNet-INTERACTION/compute_v22_map_geometry_cache.py \
  --root "$DATA_ROOT" \
  --output_dir "$DATA_ROOT/v22_map_geometry_v1"
```

The released Style statistics should be used for exact paper reproduction.
To recompute them from the training split instead:

```bash
python NTSPNet-INTERACTION/compute_style_statistics.py \
  --root "$DATA_ROOT" \
  --motion_source csv \
  --csv_motion_cache_dir "$DATA_ROOT/csv_motion_v1" \
  --output "$DATA_ROOT/train_style_statistics.pt"
```

Expected processed counts are 47,584 training scenarios and 11,794 validation
scenarios.

## Released weights

Checkpoint binaries are not committed to Git. Download them from the
`v1.0.2` GitHub Release into any local directory and verify them:

```bash
sha256sum -c SHA256SUMS
```

A complete NTSPNet or ablation checkpoint is self-contained for validation.
Training the final model or an ablation from the released initialization
requires both:

- `hpnet_local_baseline_v22.ckpt`: frozen primary HPNet trunk initialization.
- `semantic_provider_v14.ckpt`: frozen semantic-provider initialization.

See [the checkpoint guide](NTSPNet-INTERACTION/checkpoints/README.md).

## Validate a released checkpoint

```bash
export DATA_ROOT=/path/to/INTERACTION
export CHECKPOINT=/path/to/ntspnet_final_seed2048.ckpt
bash NTSPNet-INTERACTION/scripts/validate.sh
```

The program prints the scenario count and headline minJointADE, minJointFDE,
and minJointMR after validation. Cache paths and worker count can be overridden
with `CSV_MOTION_CACHE_DIR`, `V22_MAP_GEOMETRY_CACHE_DIR`, and
`NUM_WORKERS`.

## Train the final model

The released recipe uses one GPU, physical batch size 2, gradient accumulation
8 (effective batch size 16), one full-data epoch, learning rate `2e-4`, and
seed 2048.

```bash
export DATA_ROOT=/path/to/INTERACTION
export BASE_WEIGHTS=/path/to/hpnet_local_baseline_v22.ckpt
export PROVIDER_WEIGHTS=/path/to/semantic_provider_v14.ckpt
bash NTSPNet-INTERACTION/scripts/train_final.sh
```

Set `GPU_ID`, `NUM_WORKERS`, `PREFETCH_FACTOR`, or `RUN_LOG_ROOT` as
needed. Existing log files are never overwritten.

Additional full-model seeds:

```bash
bash NTSPNet-INTERACTION/scripts/train_full.sh all 1024
bash NTSPNet-INTERACTION/scripts/train_full.sh all 4096
```

## Run the ablations

With the same three environment variables set:

```bash
bash NTSPNet-INTERACTION/scripts/ablations/train_style_only.sh
bash NTSPNet-INTERACTION/scripts/ablations/train_reachability_only.sh
bash NTSPNet-INTERACTION/scripts/ablations/train_priority_only.sh
bash NTSPNet-INTERACTION/scripts/ablations/train_style_reachability.sh
bash NTSPNet-INTERACTION/scripts/ablations/train_style_priority.sh
bash NTSPNet-INTERACTION/scripts/ablations/train_reachability_priority.sh
```

These wrappers call the same training entry point and differ only in the
enabled residual module gates. This is the configuration encoded by each
released ablation checkpoint; the checkpoint filename alone does not select
modules.

## Verify the repository

Run the complete source and regression checks from the repository root:

```bash
bash NTSPNet-INTERACTION/scripts/check_reproduction.sh
```

To include a strict state-dict load of any released final or ablation
checkpoint:

```bash
export CHECKPOINT=/path/to/a/released.ckpt
bash NTSPNet-INTERACTION/scripts/check_reproduction.sh
```

This check does not train the model or run full validation. Full validation is
performed by `scripts/validate.sh` after the official validation data and
derived caches have been prepared.

## Repository layout

```text
NTSPNet-INTERACTION/
|-- model/PNet.py                 # Lightning model and prediction/loss logic
|-- modules/                      # HPNet trunk and NTSPNet semantic modules
|-- datasets/                     # INTERACTION preprocessing
|-- datamodules/                  # training/validation data pipelines
|-- transforms/                   # map, association, and augmentation transforms
|-- scripts/                      # portable final and ablation recipes
|-- configs/                      # released checkpoint manifest
|-- reproducibility_assets/       # unarchived paper-specific map semantics
|-- tests/                        # regression/unit tests
|-- train.py
|-- val.py
`-- test.py
```

Local datasets, processed files, caches, Lightning logs, generated analysis,
and checkpoint binaries are excluded by `.gitignore`.

## Acknowledgement and license

NTSPNet is derived from the official
[HPNet](https://github.com/XiaolongTang23/HPNet) implementation. The retained
baseline code and this derivative repository are distributed under the Apache
License 2.0; see [`LICENSE`](LICENSE) and
[`NTSPNet-INTERACTION/NOTICE.md`](NTSPNet-INTERACTION/NOTICE.md).

Please also follow the INTERACTION dataset license and citation requirements.
