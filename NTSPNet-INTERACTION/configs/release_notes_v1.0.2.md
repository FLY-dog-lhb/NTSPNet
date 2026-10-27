# NTSPNet v1.0.2 unarchived reproducibility assets

This release makes the exact paper map-semantic representation available as
18 individually tracked `.pt` files in:

```text
NTSPNet-INTERACTION/reproducibility_assets/v22_map_geometry_v1/
```

Each file is visible and individually downloadable from GitHub. The directory
contains its own `SHA256SUMS`, and the one-command preparation script copies
and verifies these files without requiring a compressed map archive.

## Additional changes

- Updated README, DATA.md, and checkpoint instructions to reference v1.0.2.
- Updated `prepare_paper_data.sh` to install the tracked map files.
- Updated `check_reproduction.sh` to require exactly 18 map files and verify
  every map checksum before running the regression suite.
- Removed the map archive from the v1.0.2 binary-asset set. The historical
  v1.0-paper and v1.0.1 releases remain unchanged.

## Compatibility

There is no model architecture, loss, training recipe, checkpoint, or reported
metric change. Existing checkpoints remain byte-identical. The seed-2048 final
checkpoint still loads with `strict=True` and retains SHA-256:

```text
021a3dd8b67ef180cb2b5fcb04d0bd2333322344b1860fc2710b3dc7d64f5eb8
```

Validation result over 11,794 scenarios:

```text
val_minJointADE=0.16599056
val_minJointFDE=0.53593224
val_minJointMR=0.11785654
```

Attach the 11 checkpoints, `train_style_statistics.pt`, and the v1.0.2
`SHA256SUMS` to this Release. GitHub automatically adds source archives, while
all source and map-semantic files remain directly browsable in the repository.
