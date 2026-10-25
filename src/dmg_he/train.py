"""Step 5 - one lightGBM classifier per marker, feature extractor and fold.

For fold k the classifier is trained on the training-block cores outside fold k,
features are standard-scaled with the training-set statistics, hyperparameters
are tuned by grid search with an inner ``inner_cv``-fold CV (ROC AUC), and the
refitted best model predicts the held-out fold ("val") and the complete test
block ("test"). ``fold = full`` trains on all training-block tiles instead.
"""
from __future__ import annotations

import itertools
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import features as feats
from .config import Config, fold_name
from .metrics import binary_metrics
from .splits import load_folds
from .utils import log, require, take_shard, write_csv


def binary_labels(labels: pd.DataFrame, marker: str, min_positive: int) -> pd.Series:
    """1 if the tile holds >= min_positive positive cells, 0 otherwise, NaN if the
    marker has no threshold for that core (such tiles are excluded)."""
    counts = labels[marker]
    return (counts >= min_positive).astype(float).where(counts.notna())


def make_classifier(cfg: Config) -> GridSearchCV:
    from lightgbm import LGBMClassifier

    c = cfg["classifier"]
    lgbm = LGBMClassifier(class_weight="balanced", random_state=c["random_state"], n_jobs=1,
                          device=c.get("device", "cpu"), verbose=-1)
    pipe = Pipeline([("scaler", StandardScaler()), ("lgbm", lgbm)])
    grid = {f"lgbm__{k}": v for k, v in c["param_grid"].items()}
    return GridSearchCV(pipe, grid, cv=c["inner_cv"], scoring=c["scoring"], n_jobs=c["n_jobs"], refit=True)


def _predictions(model, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"y_true": y.astype(int), "prob": model.predict_proba(X)[:, 1]}, index=X.index)


def train_one(cfg: Config, model_name: str, marker: str, fold, data: dict, overwrite: bool = False) -> None:
    paths = cfg.paths
    out = paths.model_dir(model_name, marker, fold)
    if (out / "metrics.json").exists() and not overwrite:
        log.info("[%s/%s/%s] done already - skipping", model_name, marker, fold_name(fold))
        return
    out.mkdir(parents=True, exist_ok=True)
    min_pos = cfg.tiling["min_positive_cells"]
    lab_tr, lab_te = data["labels_train"], data["labels_test"]
    y_all = binary_labels(lab_tr, marker, min_pos)
    y_test = binary_labels(lab_te, marker, min_pos).dropna()

    if str(fold) == "full":
        train_tiles, val_tiles = lab_tr.index, lab_tr.index[:0]
    else:
        f = data["folds"]["folds"][int(fold)]
        train_tiles = lab_tr.index[lab_tr.core_id.isin(f["train_cores"])]
        val_tiles = lab_tr.index[lab_tr.core_id.isin(f["val_cores"])]
    y_train = y_all.loc[train_tiles].dropna()
    y_val = y_all.loc[val_tiles].dropna()
    if y_train.nunique() < 2:
        log.warning("[%s/%s/%s] only one class in training data - skipped", model_name, marker, fold_name(fold))
        return

    X_train = data["X_train"].loc[y_train.index]
    log.info("[%s/%s/%s] training on %d tiles (%.1f%% positive)", model_name, marker, fold_name(fold),
             len(y_train), 100 * y_train.mean())
    t0 = time.time()
    search = make_classifier(cfg).fit(X_train, y_train.astype(int))
    best = search.best_estimator_
    joblib.dump(best, out / "model.joblib")

    thr = cfg["evaluation"]["threshold"]
    result = {"model": model_name, "marker": marker, "fold": fold_name(fold),
              "best_params": {k.removeprefix("lgbm__"): v for k, v in search.best_params_.items()},
              "inner_cv_score": float(search.best_score_), "fit_seconds": round(time.time() - t0, 1)}
    if len(y_val):
        pv = _predictions(best, data["X_train"].loc[y_val.index], y_val)
        write_csv(pv, out / "predictions_val.csv")
        result["val"] = binary_metrics(pv.y_true, pv.prob, thr)
    pt = _predictions(best, data["X_test"].loc[y_test.index], y_test)
    write_csv(pt, out / "predictions_test.csv")
    result["test"] = binary_metrics(pt.y_true, pt.prob, thr)
    (out / "metrics.json").write_text(json.dumps(result, indent=1, default=float))
    log.info("[%s/%s/%s] val ROC AUC %.3f | test ROC AUC %.3f", model_name, marker, fold_name(fold),
             result.get("val", {}).get("roc_auc", np.nan), result["test"]["roc_auc"])


def parse_folds(cfg: Config, folds) -> list:
    n = cfg["cv"]["n_folds"]
    out = []
    for f in folds or ["all"]:
        if f == "all":
            out += list(range(n))
        elif f == "full":
            out.append("full")
        else:
            out.append(int(f))
    return out


def parse_markers(cfg: Config, markers) -> list[str]:
    if not markers or markers == ["all"]:
        return cfg.marker_names
    unknown = set(markers) - set(cfg.marker_names)
    if unknown:
        raise KeyError(f"unknown marker(s) {sorted(unknown)}; available: {cfg.marker_names}")
    return list(markers)


def run(cfg: Config, model: str, markers=None, folds=None, shard: str | None = None,
        overwrite: bool = False) -> None:
    jobs = list(itertools.product(parse_markers(cfg, markers), parse_folds(cfg, folds)))
    jobs = take_shard(jobs, shard)
    log.info("[%s] %d marker/fold combinations to run", model, len(jobs))
    if not jobs:
        return
    paths = cfg.paths
    data = {
        "labels_train": pd.read_csv(require(paths.merged_labels("train"), "split"), index_col=0),
        "labels_test": pd.read_csv(require(paths.merged_labels("test"), "split"), index_col=0),
        "X_train": feats.load(cfg, model, "train"),
        "X_test": feats.load(cfg, model, "test"),
        "folds": load_folds(cfg),
    }
    for marker, fold in jobs:
        train_one(cfg, model, marker, fold, data, overwrite=overwrite)
