# NTSPNet v1.0-paper

This release contains the checkpoints and small preprocessing assets used for
the INTERACTION validation results reported in the repository README.

## Headline result

The seed-2048 final model obtains:

- minJointADE: 0.16599056
- minJointFDE: 0.53593224
- minJointMR: 0.11785654
- validation scenarios: 11,794

## Assets

- Final NTSPNet checkpoints for seeds 1024, 2048, and 4096.
- Six single/pairwise module ablation checkpoints.
- Local HPNet V22 baseline and V14 semantic-provider initialization weights.
- Released training-only Driving Style statistics.
- V22 map-geometry cache used by the reported experiments.
- SHA256SUMS for integrity verification.

Checkpoint roles and exact metrics are documented in
`NTSPNet-INTERACTION/configs/released_checkpoints.csv`.

The CSV motion cache is not included because it can be regenerated from the
official INTERACTION CSV files with `compute_csv_motion_cache.py`. The
INTERACTION dataset itself is not redistributed.
