# Differences from the original analysis scripts

This package replaces the per-model scripts and notebooks used during the analysis
(`preprocessData_2_*.py`, `get_feature_*.py`, `predictXGB_*.py`, notebooks 1–6). The
method is the same; the points below changed and can lead to small numerical differences
when results are regenerated.

| Topic | Original scripts | This package |
|---|---|---|
| Tile size | several configurations (25, 50, 100 µm) selected via file-name suffixes | 100 µm only, set in the config |
| Tile/feature alignment | RetCCL, CTransPath, REMEDIS, Virchow and Prov-GigaPath scripts prepended each new batch (`pd.concat([new, old])`) and then assigned tile names in forward order, so embeddings were assigned to the wrong tiles. UNI was correct. | embeddings are stored in tile order, indexed by tile ID, and checked against the label table |
| Minimum cells per tile | `n_cells > min_cells` (i.e. ≥ 6 for `min_cells = 5`) | `n_cells >= min_cells_per_tile` (≥ 5, as stated in the Methods) |
| Marker without threshold for a core | tile counted as negative | tile excluded for that marker |
| DMG definition | split stratified with DIPG / DMG spinal / DMG Thalamic; subgroup analysis additionally included DMG Medulla / DMG cerebellum | DIPG / DMG spinal / DMG Thalamic (`dmg_diagnoses`) for both the split and the subgroups |
| Feature scaling during grid search | scaler fit once on the whole training fold before the inner CV | scaler is part of the pipeline and refit within each inner CV fold |
| Threshold-dependent metrics | threshold at 90 % recall in the prediction scripts; 0.5 in a separate notebook | probability threshold 0.5 (configurable) |
| CTransPath weights | loaded with `strict=False` without checking | missing weights raise an error |
| REMEDIS | called PyTorch methods (`.eval()`, `.to()`) on a TensorFlow model | TensorFlow-only, batched |
| "skip finished runs" check | never skipped (`sys.exit` was caught by a bare `except`) | finished marker/fold runs are skipped unless `--overwrite` |
| Split file | Python pickle | `splits.json` (plus a summary table) |
| Tile images | RGBA PNG + `.npy` copy | RGB PNG |
| Iba1 exception for threshold 2116 | switched to the whole-cell column, which was already used | removed (no effect) |
