"""Create a small synthetic dataset with the same file formats as the study data."""
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import yaml

RES = 0.325  # um per pixel


def _block(rng, markers, block_id, n_patients, cores_per_patient, core_px, gap=100):
    n_cores = n_patients * cores_per_patient
    ncol = 4
    nrow = int(np.ceil(n_cores / ncol))
    slide = np.full((nrow * (core_px + gap) + gap, ncol * (core_px + gap) + gap, 3), 240, np.uint8)
    cell_tables, thresholds, meta = [], {}, []
    for k in range(n_cores):
        core = f"{chr(65 + k // 9)}-{k % 9 + 1}"
        patient = k // cores_per_patient
        dmg = patient % 2 == 0
        meta.append({"Name": core, "TMA": block_id, "PID": f"P{block_id}_{patient}",
                     "Diagnosis": "DIPG" if dmg else ("HGG" if k % 3 else "CNTRL"),
                     "Dissemination": "Primary" if k % 4 < 2 else "Metastasis"})
        oy, ox = gap + (k // ncol) * (core_px + gap), gap + (k % ncol) * (core_px + gap)
        n = 400
        xs, ys = ox + rng.uniform(0, core_px, n), oy + rng.uniform(0, core_px, n)
        # H3K27M-positive cells in the left half of DMG cores, painted red on the slide
        pos = dmg & (xs < ox + core_px / 2)
        region = np.zeros((core_px, core_px, 3), int) + (200, 150, 200)
        if dmg:
            region[:, : core_px // 2] = (220, 60, 90)
        region += rng.integers(-15, 15, region.shape)
        slide[oy:oy + core_px, ox:ox + core_px] = np.clip(region, 0, 255)
        cols = {"TMA core": core, "Centroid X Âµm": xs * RES, "Centroid Y Âµm": ys * RES}
        for m, spec in markers.items():
            vals = rng.lognormal(4.5, 0.6, n)
            if m == "H3K27M":
                vals = np.where(pos, 2000.0, 50.0) + rng.normal(0, 10, n)
            cols[f"{spec['compartment']}: {m} mean"] = vals
        cell_tables.append(pd.DataFrame(cols))
        thresholds[core] = {spec["threshold_name"]: 400.0 for spec in markers.values()}
    cells = pd.concat(cell_tables, ignore_index=True)
    names = [spec["threshold_name"] for spec in markers.values()]
    thr = pd.DataFrame([names] + [[thresholds[c][x] for x in names] for c in thresholds],
                       index=["Name", *thresholds], columns=[f"Mean_{i}" for i in range(len(names))])
    return cells, thr, slide, meta


def make(root: Path, n_train_patients=6, n_test_patients=4, cores_per_patient=2, core_px=900, seed=0) -> Path:
    from dmg_he.config import Config

    rng = np.random.default_rng(seed)
    markers = Config.load().markers
    data = Path(root) / "data"
    data.mkdir(parents=True, exist_ok=True)
    meta = []
    for block_id, n_pat in ((272, n_train_patients), (273, n_test_patients)):
        cells, thr, slide, m = _block(rng, markers, block_id, n_pat, cores_per_patient, core_px)
        cells.to_csv(data / f"cells_{block_id}.csv")
        thr.to_csv(data / f"thr_{block_id}.csv")
        tifffile.imwrite(data / f"slide_{block_id}.tif", slide)
        meta += m
    pd.DataFrame(meta).to_csv(data / "meta.csv", index=False)

    cfg = {
        "data_dir": "data", "work_dir": "work",
        "blocks": {
            "train": {"id": 272, "cells": "cells_272.csv", "thresholds": "thr_272.csv", "slide": "slide_272.tif"},
            "test": {"id": 273, "cells": "cells_273.csv", "thresholds": "thr_273.csv", "slide": "slide_273.tif"},
        },
        "metadata": "meta.csv",
        "cv": {"n_folds": 3},
        "classifier": {"n_jobs": 1, "inner_cv": 2,
                       "param_grid": {"num_leaves": [5], "learning_rate": [0.1], "n_estimators": [20, 50]}},
        "feature_extractors": {"batch_size": 16},
    }
    path = Path(root) / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path
