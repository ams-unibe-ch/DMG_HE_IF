"""Step 6 - performance summaries, model comparison and spatial prediction maps."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, fold_name
from .metrics import binary_metrics
from .utils import log, require, write_csv


# ------------------------------------------------------------------ helpers
def subgroup_cores(cfg: Config) -> dict[str, list[str]]:
    """Test-block cores per subgroup defined in ``evaluation.subgroups``."""
    meta = pd.read_csv(require(cfg.paths.core_metadata("test"), "prepare"), index_col=0)
    groups = {"all": meta.index.to_list()}
    for name, filters in (cfg["evaluation"].get("subgroups") or {}).items():
        keep = pd.Series(True, index=meta.index)
        for col, allowed in filters.items():
            if allowed == "dmg":
                allowed = cfg["dmg_diagnoses"]
            keep &= meta[col].astype(str).str.strip().isin([str(a) for a in allowed])
        groups[name] = meta.index[keep].to_list()
    return groups


def _folds_available(cfg, model, marker):
    folds = [k for k in range(cfg["cv"]["n_folds"])] + ["full"]
    return [f for f in folds if (cfg.paths.model_dir(model, marker, f) / "predictions_test.csv").exists()]


def per_fold_metrics(cfg: Config, model: str) -> pd.DataFrame:
    """Metrics for every marker x fold x evaluation set, recomputed from the stored predictions."""
    thr = cfg["evaluation"]["threshold"]
    groups = subgroup_cores(cfg)
    test_core = pd.read_csv(cfg.paths.merged_labels("test"), index_col=0, usecols=["tile_id", "core_id"]).core_id
    rows = []
    for marker in cfg.marker_names:
        for fold in _folds_available(cfg, model, marker):
            d = cfg.paths.model_dir(model, marker, fold)
            sets = {}
            if (d / "predictions_val.csv").exists():
                sets["val"] = pd.read_csv(d / "predictions_val.csv", index_col=0)
            pt = pd.read_csv(d / "predictions_test.csv", index_col=0)
            cores = test_core.reindex(pt.index)
            for g, members in groups.items():
                sets["test" if g == "all" else f"test_{g}"] = pt[cores.isin(members).to_numpy()]
            for name, p in sets.items():
                rows.append({"model": model, "marker": marker, "fold": fold_name(fold), "set": name,
                             **binary_metrics(p.y_true, p.prob, thr)})
    if not rows:
        raise FileNotFoundError(f"no predictions found for model {model} - run `dmg-he train --model {model}`")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD over the cross-validation folds (the 'full' model is reported separately)."""
    cv = df[df.fold != "full"]
    num = [c for c in cv.columns if c not in ("model", "marker", "fold", "set")]
    agg = cv.groupby(["model", "set", "marker"])[num].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_folds"] = cv.groupby(["model", "set", "marker"]).size()
    return agg.reset_index()


# -------------------------------------------------------------------- plots
def _bar(ax, summary, metric, random_level, title):
    s = summary.sort_values(f"{metric}_mean", ascending=False)
    x = np.arange(len(s))
    ax.bar(x, s[f"{metric}_mean"], yerr=s[f"{metric}_std"], capsize=3, color="#9aa5b1")
    ax.axhline(random_level, color="#c0392b", ls="--", lw=1, label="random classifier")
    ax.set_xticks(x, s.marker, rotation=90)
    ax.set_ylabel({"roc_auc": "ROC AUC", "rel_auprc": "relative AUPRC"}.get(metric, metric))
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8)


def plot_summary(summary: pd.DataFrame, out_dir, model: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for set_name, s in summary.groupby("set"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        _bar(axes[0], s, "roc_auc", 0.5, f"{model} - {set_name}")
        _bar(axes[1], s, "rel_auprc", 1.0, f"{model} - {set_name}")
        fig.tight_layout()
        fig.savefig(out_dir / f"barplot_{set_name}.png", dpi=150)
        plt.close(fig)


def plot_comparison(summary: pd.DataFrame, out_path, set_name="test", metric="roc_auc") -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summary[summary.set == set_name]
    mean = s.pivot(index="marker", columns="model", values=f"{metric}_mean")
    std = s.pivot(index="marker", columns="model", values=f"{metric}_std")
    order = mean.max(axis=1).sort_values(ascending=False).index
    mean, std = mean.loc[order], std.loc[order]
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(mean)), 3.8))
    w = 0.8 / max(1, mean.shape[1])
    x = np.arange(len(mean))
    for j, m in enumerate(mean.columns):
        ax.bar(x + (j - (mean.shape[1] - 1) / 2) * w, mean[m], w, yerr=std[m], capsize=1.5, label=m)
    ax.axhline(0.5 if metric == "roc_auc" else 1.0, color="#c0392b", ls="--", lw=1)
    ax.set_xticks(x, mean.index, rotation=90)
    ax.set_ylabel("ROC AUC" if metric == "roc_auc" else metric)
    ax.set_ylim(0.4 if metric == "roc_auc" else 0, None)
    ax.legend(ncol=min(6, mean.shape[1]), fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return mean


def run(cfg: Config, models: list[str]) -> None:
    summaries = []
    for model in models:
        out = cfg.paths.results_dir(model)
        out.mkdir(parents=True, exist_ok=True)
        df = per_fold_metrics(cfg, model)
        write_csv(df, out / "metrics_per_fold.csv", index=False)
        summary = summarize(df)
        write_csv(summary, out / "metrics_summary.csv", index=False)
        if (df.fold == "full").any():
            write_csv(df[df.fold == "full"], out / "metrics_full_model.csv", index=False)
        plot_summary(summary, out, model)
        summaries.append(summary)
        t = summary[summary.set == "test"]
        log.info("[%s] test ROC AUC (mean over folds) %.2f-%.2f -> %s", model,
                 t.roc_auc_mean.min(), t.roc_auc_mean.max(), out)
    if len(models) > 1:
        out = cfg.paths.root / "results" / "comparison"
        out.mkdir(parents=True, exist_ok=True)
        allsum = pd.concat(summaries)
        write_csv(allsum, out / "metrics_summary_all_models.csv", index=False)
        for metric in ("roc_auc", "rel_auprc"):
            table = plot_comparison(allsum, out / f"comparison_test_{metric}.png", metric=metric)
            write_csv(table, out / f"comparison_test_{metric}.csv")
        log.info("model comparison written to %s", out)


# ------------------------------------------------------------- spatial maps
def plot_core(cfg: Config, model: str, marker: str, core: str, out_path=None) -> str:
    """Core image, IF-derived tile labels, predicted probability (mean over the
    CV models) and difference map for one test-block core."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    paths = cfg.paths
    meta = pd.read_csv(paths.core_metadata("test"), index_col=0)
    if core not in meta.index:
        alt = f"{core}_{cfg['blocks']['test']['id']}"
        if alt not in meta.index:
            raise KeyError(f"core {core!r} not in the test block")
        core = alt
    labels = pd.read_csv(paths.tile_labels("test", core), index_col="tile_id")
    probs = []
    for k in range(cfg["cv"]["n_folds"]):
        f = paths.model_dir(model, marker, k) / "predictions_test.csv"
        if f.exists():
            probs.append(pd.read_csv(f, index_col=0).prob.reindex(labels.index))
    if not probs:
        raise FileNotFoundError(f"no test predictions for {model}/{marker}")
    prob = pd.concat(probs, axis=1).mean(axis=1)
    true = (labels[marker] >= cfg.tiling["min_positive_cells"]).astype(float).where(labels[marker].notna())
    img = plt.imread(require(paths.core_image("test", core), "tile"))
    x0c, y0c = int(meta.at[core, "x_min"]), int(meta.at[core, "y_min"])
    side = cfg.tile_size_px
    cmap = LinearSegmentedColormap.from_list("bwr", [(0, "blue"), (0.5, "white"), (1, "red")])

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    panels = [("virtual H&E", None, (0, 1)), ("true label", true, (0, 1)),
              ("predicted probability", prob, (0, 1)), ("true - predicted", true - prob, (-1, 1))]
    for ax, (title, values, vrange) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
        if values is None:
            continue
        # overlapping tiles: every pixel shows the mean value of the tiles covering it
        total, count = np.zeros(img.shape[:2]), np.zeros(img.shape[:2])
        for tid, v in values.dropna().items():
            x, y = int(labels.at[tid, "x0"]) - x0c, int(labels.at[tid, "y0"]) - y0c
            total[y:y + side, x:x + side] += v
            count[y:y + side, x:x + side] += 1
        with np.errstate(invalid="ignore"):
            raster = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        im = ax.imshow(raster, cmap=cmap, vmin=vrange[0], vmax=vrange[1], alpha=0.55, interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{core} - {marker} - {model}", fontsize=10)
    fig.tight_layout()
    if out_path is None:
        out_dir = paths.results_dir(model) / "spatial_maps"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{core}_{marker}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("spatial map written to %s", out_path)
    return str(out_path)
