"""Performance metrics for binary tile predictions."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)

METRICS = ["n_tiles", "prevalence", "roc_auc", "auprc", "rel_auprc",
           "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1"]


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Threshold-free metrics (ROC AUC, AUPRC, AUPRC / prevalence) and
    threshold-dependent metrics at ``threshold``. Undefined values are NaN."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    n, prev = len(y_true), float(y_true.mean()) if len(y_true) else np.nan
    if n == 0:
        return {m: (0 if m == "n_tiles" else np.nan) for m in METRICS}
    both = len(np.unique(y_true)) == 2
    out = {"n_tiles": n, "prevalence": prev}
    out["roc_auc"] = roc_auc_score(y_true, y_prob) if both else np.nan
    out["auprc"] = average_precision_score(y_true, y_prob) if y_true.sum() > 0 else np.nan
    out["rel_auprc"] = out["auprc"] / prev if prev and prev > 0 else np.nan
    out["accuracy"] = accuracy_score(y_true, y_pred) if n else np.nan
    out["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred) if both else np.nan
    out["precision"] = precision_score(y_true, y_pred, zero_division=0)
    out["recall"] = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["specificity"] = tn / (tn + fp) if (tn + fp) else np.nan
    out["f1"] = f1_score(y_true, y_pred, zero_division=0)
    return out
