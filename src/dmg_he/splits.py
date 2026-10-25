"""Step 3 - merge tile labels and define the cross-validation folds.

Cores of the training block are split into ``n_folds`` folds with
``StratifiedGroupKFold``: all cores of a patient fall into the same fold and the
folds are stratified by DMG / non-DMG core.
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import Config
from .utils import log, require, write_csv


def merge_labels(cfg: Config, role: str) -> pd.DataFrame:
    paths = cfg.paths
    meta = pd.read_csv(require(paths.core_metadata(role), "prepare"), index_col=0)
    frames, missing = [], []
    for core in meta.index:
        f = paths.tile_labels(role, core)
        if f.exists():
            frames.append(pd.read_csv(f))
        else:
            missing.append(core)
    if missing:
        raise FileNotFoundError(f"[{role}] tile labels missing for {len(missing)} cores "
                                f"(e.g. {missing[:3]}) - run `dmg-he tile` first")
    labels = pd.concat(frames, ignore_index=True).set_index("tile_id")
    write_csv(labels, paths.merged_labels(role))
    n_tiles = labels.groupby("core_id").size()
    meta["n_tiles"] = n_tiles.reindex(meta.index).fillna(0).astype(int)
    write_csv(meta, paths.core_metadata(role))
    log.info("[%s] %d tiles from %d cores (%d cores without tiles)", role, len(labels),
             int((meta.n_tiles > 0).sum()), int((meta.n_tiles == 0).sum()))
    return labels


def make_folds(cfg: Config) -> dict:
    meta = pd.read_csv(cfg.paths.core_metadata("train"), index_col=0)
    meta = meta[meta.n_tiles > 0]
    if meta.patient.isna().any():
        raise ValueError("some training cores have no patient ID in the metadata: "
                         f"{meta.index[meta.patient.isna()].to_list()[:5]}")
    n = cfg["cv"]["n_folds"]
    sgkf = StratifiedGroupKFold(n_splits=n)
    folds, summary = [], []
    for k, (tr, va) in enumerate(sgkf.split(meta, meta.is_DMG.astype(int), meta.patient)):
        tr_cores, va_cores = meta.index[tr].to_list(), meta.index[va].to_list()
        folds.append({"fold": k, "train_cores": tr_cores, "val_cores": va_cores})
        summary.append({
            "fold": k,
            "n_cores_train": len(tr), "n_cores_val": len(va),
            "n_patients_train": meta.patient.iloc[tr].nunique(),
            "n_patients_val": meta.patient.iloc[va].nunique(),
            "n_tiles_train": int(meta.n_tiles.iloc[tr].sum()),
            "n_tiles_val": int(meta.n_tiles.iloc[va].sum()),
            "frac_DMG_cores_train": meta.is_DMG.iloc[tr].mean(),
            "frac_DMG_cores_val": meta.is_DMG.iloc[va].mean(),
        })
    out = {"n_folds": n, "folds": folds}
    paths = cfg.paths
    paths.splits.parent.mkdir(parents=True, exist_ok=True)
    paths.splits.write_text(json.dumps(out, indent=1))
    summary = pd.DataFrame(summary).set_index("fold")
    write_csv(summary, paths.splits.with_name("splits_summary.csv"))
    log.info("cross-validation folds written to %s\n%s", paths.splits, summary.to_string())
    return out


def load_folds(cfg: Config) -> dict:
    return json.loads(require(cfg.paths.splits, "split").read_text())


def run(cfg: Config) -> None:
    for role in cfg.roles:
        merge_labels(cfg, role)
    make_folds(cfg)
