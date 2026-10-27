# V22 map geometry for exact NTSPNet reproduction

This directory contains the 18 map-specific geometry sidecars used by every
reported NTSPNet validation, full-data, random-seed, and ablation result.
Files are stored individually so they can be browsed and downloaded without
extracting an archive.

Each filename corresponds to one INTERACTION map. The sidecars preserve the
map geometry semantics expected by the frozen local HPNet V22 prediction
trunk. They are generated from the official INTERACTION maps by
`compute_v22_map_geometry_cache.py`.

Verify all files from this directory:

```bash
sha256sum -c SHA256SUMS
```

The public preparation entry point
`../../scripts/prepare_paper_data.sh` verifies these hashes before copying the
directory into an INTERACTION dataset root.
