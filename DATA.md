# INTERACTION data for NTSPNet

NTSPNet is trained and evaluated on the INTERACTION dataset. The raw dataset is
not copied into this Git repository. Download it from the
[official INTERACTION Hugging Face page](https://huggingface.co/datasets/INTERACTION-dataset/INTERACTION-dataset)
and follow its CC BY 4.0 license and citation requirements. The dataset
maintainers state that the previous website is being sunset in favor of that
page.

The official support code is maintained separately at
[interaction-dataset/interaction-dataset](https://github.com/interaction-dataset/interaction-dataset).

## Required layout

NTSPNet expects the extracted files in this layout:

```text
/path/to/INTERACTION/
|-- maps/
|   `-- *.osm
|-- train/
|   `-- *.csv
|-- val/
|   `-- *.csv
`-- test_multi-agent/          # optional; required only for challenge export
    `-- *.csv
```

Set the dataset root before running repository scripts:

```bash
export DATA_ROOT=/path/to/INTERACTION
```

The released paper experiments use 10 observed steps and 30 future steps.
After preprocessing, the expected counts are:

| Split | Scenarios |
|---|---:|
| train | 47,584 |
| validation | 11,794 |

A count mismatch means that the extracted raw release or preprocessing setup is
not the one used for the reported results.

## Derived data used by the paper

| Artifact | Source | Reproduction rule |
|---|---|---|
| `train_processed/`, `val_processed/` | official CSV and map files | regenerate locally |
| `csv_motion_v1/train/`, `csv_motion_v1/val/` | official CSV files | regenerate locally |
| `v22_map_geometry_v1/*.pt` | official map files | copy the 18 individually tracked repository files |
| `train_style_statistics.pt` | official training split | download the released file |

The exact V22 map semantics are stored without an archive at:

```text
NTSPNet-INTERACTION/reproducibility_assets/v22_map_geometry_v1/
```

The directory contains 18 `.pt` files and its own `SHA256SUMS`. Each file can
be inspected and downloaded separately from GitHub. The Style statistics remain
a single file attached to the GitHub Release and are covered by the Release
`SHA256SUMS`.

The processed scenarios and CSV-motion cache are intentionally not distributed:
they are deterministic derivatives of the official raw dataset and are too large
for the source repository.

## One-command preparation

Download `train_style_statistics.pt` and `SHA256SUMS` from the v1.0.2 Release
into one directory, then run from the repository root:

```bash
export DATA_ROOT=/path/to/INTERACTION
export PAPER_ASSET_DIR=/path/to/downloaded/release/assets
bash NTSPNet-INTERACTION/scripts/prepare_paper_data.sh
```

The script verifies the official raw directories, all 18 tracked map files, and
the released Style-statistics file. It then creates the processed scenarios,
generates the CSV-motion cache, installs both paper-specific assets without
overwriting existing copies, and checks the final scenario and map counts.

To perform the steps manually, use the commands in the root
[`README.md`](README.md).

## Validation-only reproduction

A complete final or ablation checkpoint is self-contained. After preparing the
validation split and caches:

```bash
export DATA_ROOT=/path/to/INTERACTION
export CHECKPOINT=/path/to/a/released.ckpt
bash NTSPNet-INTERACTION/scripts/validate.sh
```

The V22 baseline and V14 semantic-provider checkpoints are needed only when
training a new final or ablation model.

## Integrity and provenance

Verify the unarchived map assets directly in the repository:

```bash
cd NTSPNet-INTERACTION/reproducibility_assets/v22_map_geometry_v1
sha256sum -c SHA256SUMS
```

Verify downloaded Release assets from their download directory:

```bash
cd "$PAPER_ASSET_DIR"
sha256sum -c SHA256SUMS
```

Keep the official INTERACTION archive names and a record of the download date.
Do not commit raw INTERACTION files, generated processed scenarios, or the
CSV-motion cache. This separation preserves dataset licensing and provenance
while keeping every NTSPNet-specific representation, recipe, checksum, weight,
and result manifest publicly accessible from the repository and its Release.
