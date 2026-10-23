# Released checkpoints

Checkpoint binaries are intentionally not stored in Git. Download them from
the `v1.0-paper` GitHub Release. The exact filenames, roles, sizes, SHA-256
hashes, and validation metrics are recorded in
`../configs/released_checkpoints.csv`.

From the directory containing the downloaded assets, verify every file with:

```bash
sha256sum -c SHA256SUMS
```

For validation, a complete final or ablation checkpoint is self-contained.
The V22 and V14 files are not loaded by `val.py`.

For new final/ablation training, both initialization files are required:

- `hpnet_local_baseline_v22.ckpt`: frozen primary prediction trunk.
- `semantic_provider_v14.ckpt`: frozen semantic context provider.

Exact paper reproduction also uses `train_style_statistics.pt` and the
contents of `v22_map_geometry_v1.tar.gz`. Extract the latter into the
INTERACTION dataset root. The CSV motion cache is regenerated locally from the
official raw CSV files and is not redistributed.
