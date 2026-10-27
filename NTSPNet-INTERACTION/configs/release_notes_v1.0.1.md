# NTSPNet v1.0.1 reproducibility update

This release repairs the public reproduction package after the implementation
directory was renamed from `HPNet-INTERACTION` to
`NTSPNet-INTERACTION`.

## Changes

- Updated every repository command and link to the new directory name.
- Added `DATA.md` with the official INTERACTION source, licensing,
  provenance, expected scenario counts, and derived-artifact roles.
- Added `scripts/prepare_paper_data.sh` for one-command preparation of the
  exact paper data representation.
- Added `scripts/check_reproduction.sh` for Shell syntax checks, Python
  compilation, the complete regression suite, and optional strict checkpoint
  loading.
- Published the four analysis helpers required by the included unit tests.
- Added the tested `pytest==7.4.4` dependency.

## Compatibility

There is no model architecture, loss, training recipe, checkpoint, or reported
metric change in v1.0.1. Existing v1.0-paper checkpoints remain valid. The
seed-2048 final checkpoint still loads with `strict=True` and retains SHA-256:

```text
021a3dd8b67ef180cb2b5fcb04d0bd2333322344b1860fc2710b3dc7d64f5eb8
```

Validation result over 11,794 scenarios:

```text
val_minJointADE=0.16599056
val_minJointFDE=0.53593224
val_minJointMR=0.11785654
```

## Verified repository state

The release source was checked with:

```bash
PYTHON_BIN=/path/to/the/paper/environment/bin/python \
CHECKPOINT=/path/to/ntspnet_final_seed2048.ckpt \
bash NTSPNet-INTERACTION/scripts/check_reproduction.sh
```

Result:

```text
193 passed, 11 warnings
strict checkpoint load: OK (PNet)
Repository reproduction checks: OK
```

Attach the same checkpoint and paper-data assets listed in
`configs/released_checkpoints.csv`, together with `SHA256SUMS`, to the
v1.0.1 GitHub Release so that source, recipes, checksums, weights, and
NTSPNet-specific data assets remain available from one release page. The raw
INTERACTION dataset remains distributed by its official maintainers.
