"""Step 1 - split the QuPath cell-by-cell export into per-core tables.

For every core of a TMA block this writes the cell positions (in slide pixels)
and the marker intensities, and collects one metadata row per core with the
bounding box, the per-core marker thresholds, the number of positive cells and
the clinical annotation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .utils import log, write_csv


def intensity_column(marker: str, spec: dict) -> str:
    return f"{spec['compartment']}: {marker} mean"


def read_thresholds(path) -> pd.DataFrame:
    """Read a per-core threshold table.

    Expected layout (as exported for the study): first column = core name,
    one row called ``Name`` holding the marker names, all other rows = cores.
    Returns a cores x marker-names table of floats.
    """
    df = pd.read_csv(path, index_col=0)
    if "Name" not in df.index:
        raise ValueError(f"{path}: expected a row called 'Name' with the marker names")
    df.columns = df.loc["Name"].astype(str).str.strip().to_list()
    df = df.drop(index="Name")
    df.index = df.index.astype(str).str.strip()
    return df.apply(pd.to_numeric, errors="coerce")


def _find_column(columns, wanted: str, prefix: str) -> str:
    if wanted != "auto":
        if wanted not in columns:
            raise KeyError(f"column {wanted!r} not found in cell table")
        return wanted
    for c in columns:
        if c.startswith(prefix):
            return c
    raise KeyError(f"no column starting with {prefix!r} in cell table")


def read_metadata(cfg: Config, block_id: int) -> pd.DataFrame:
    mc = cfg["metadata_columns"]
    meta = pd.read_csv(cfg.metadata_file)
    meta = meta[meta[mc["block"]] == block_id].copy()
    meta[mc["core"]] = meta[mc["core"]].astype(str).str.strip()
    for key in ("diagnosis", "dissemination"):
        col = mc.get(key)
        if col in meta:
            meta[col] = meta[col].astype(str).str.strip()
    if meta[mc["core"]].duplicated().any():
        dups = meta.loc[meta[mc["core"]].duplicated(), mc["core"]].to_list()
        raise ValueError(f"duplicate core names in metadata for block {block_id}: {dups[:5]}")
    return meta.set_index(mc["core"])


def prepare_block(cfg: Config, role: str) -> pd.DataFrame:
    block = cfg.block(role)
    paths = cfg.paths
    res = cfg.tiling["resolution_um_per_px"]
    ct = cfg["cell_table"]
    mc = cfg["metadata_columns"]
    dmg = set(cfg["dmg_diagnoses"])
    controls = set(cfg["control_diagnoses"])

    log.info("[%s] reading thresholds %s", role, block["thresholds"].name)
    thresholds = read_thresholds(block["thresholds"])

    header = pd.read_csv(block["cells"], nrows=0).columns.to_list()
    xcol = _find_column(header, ct["x_column"], "Centroid X")
    ycol = _find_column(header, ct["y_column"], "Centroid Y")
    marker_cols = {}
    for m, spec in cfg.markers.items():
        col = intensity_column(m, spec)
        if col in header:
            marker_cols[m] = col
        else:
            log.warning("[%s] intensity column %r missing - marker %s skipped", role, col, m)
    usecols = [ct["core_column"], xcol, ycol, *marker_cols.values()]
    log.info("[%s] reading cell table %s", role, block["cells"].name)
    cells = pd.read_csv(block["cells"], usecols=usecols)
    cells[ct["core_column"]] = cells[ct["core_column"]].astype(str).str.strip()
    meta = read_metadata(cfg, block["id"])

    rows = []
    groups = dict(tuple(cells.groupby(ct["core_column"], sort=False)))
    for core in thresholds.index:
        df = groups.get(core)
        if df is None or len(df) < ct["min_cells_per_core"]:
            log.debug("[%s] core %s skipped (fewer than %d cells)", role, core, ct["min_cells_per_core"])
            continue
        core_id = f"{core}_{block['id']}"
        out = pd.DataFrame({
            "x_px": (df[xcol].to_numpy() / res).astype(int),
            "y_px": (df[ycol].to_numpy() / res).astype(int),
        })
        for m, col in marker_cols.items():
            out[m] = df[col].to_numpy()
        write_csv(out, paths.core_cells(role, core_id), index=False)

        row = {
            "core_id": core_id, "core": core, "block": block["id"], "role": role,
            "x_min": out.x_px.min(), "y_min": out.y_px.min(),
            "x_max": out.x_px.max(), "y_max": out.y_px.max(),
            "n_cells": len(out),
        }
        for m in marker_cols:
            thr_name = cfg.markers[m]["threshold_name"]
            thr = thresholds.at[core, thr_name] if thr_name in thresholds.columns else np.nan
            row[f"threshold_{m}"] = thr
            row[f"n_pos_{m}"] = int((out[m] > thr).sum()) if pd.notna(thr) else np.nan
        if core in meta.index:
            info = meta.loc[core]
            diagnosis = info.get(mc["diagnosis"], "")
            row["patient"] = info.get(mc["patient"])
            row["diagnosis"] = diagnosis
            row["dissemination"] = info.get(mc["dissemination"])
            row["is_DMG"] = diagnosis in dmg
            row["is_control"] = diagnosis in controls
            for col, val in info.items():
                row.setdefault(f"meta_{col}", val)
        else:
            log.warning("[%s] core %s not found in metadata", role, core)
            row.update(patient=None, diagnosis=None, dissemination=None, is_DMG=False, is_control=False)
        rows.append(row)

    md = pd.DataFrame(rows).set_index("core_id")
    write_csv(md, paths.core_metadata(role))
    counts = md.groupby(md["diagnosis"].fillna("<missing>"))["is_DMG"].agg(["size", "first"])
    log.info("[%s] diagnoses (cores, counted as DMG):\n%s", role,
             "\n".join(f"    {d!r}: {n} cores, DMG={bool(f)}" for d, (n, f) in counts.iterrows()))
    unmatched = [d for d in counts.index if "DMG" in str(d).upper() and d not in dmg]
    if unmatched:
        log.info("[%s] note: %s contain 'DMG' but are not in dmg_diagnoses, so they are "
                 "treated as non-DMG", role, unmatched)
    log.info("[%s] %d cores written to %s", role, len(md), paths.cores_dir(role))
    missing_thr = [c for c in md.columns if c.startswith("threshold_") and md[c].isna().any()]
    if missing_thr:
        log.warning("[%s] some cores have no threshold for: %s", role,
                    ", ".join(c.removeprefix("threshold_") for c in missing_thr))
    return md


def run(cfg: Config, roles=None) -> None:
    for role in roles or cfg.roles:
        prepare_block(cfg, role)
