"""End-to-end test of the full pipeline on synthetic data (rgb_stats features)."""
import json

import numpy as np
import pandas as pd
import pytest

from dmg_he import cli
from dmg_he.config import Config
from dmg_he.tiling import tile_windows

from synthetic import make


def test_tile_windows_geometry():
    # cells spanning 0..1000 px, 307 px tiles, 153 px stride -> x0 in {0,153,306,459,612}
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 1000, 5000), rng.uniform(0, 1000, 5000)
    x[:2], y[:2] = (0, 1000), (0, 1000)
    wins = list(tile_windows(x, y, 307, 153, 5))
    starts = sorted({w[0] for w in wins})
    assert starts == [0, 153, 306, 459, 612]
    assert len(wins) == 25


def test_tile_windows_min_cells():
    x = np.array([0, 10, 20, 30, 400.0])
    y = np.array([0, 10, 20, 30, 400.0])
    assert len(list(tile_windows(x, y, 100, 50, 4))) == 1
    assert len(list(tile_windows(x, y, 100, 50, 5))) == 0


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("proj")
    cfg_path = make(root)
    run = lambda *a: cli.main([a[0], "-c", str(cfg_path), *a[1:]])  # noqa: E731
    assert run("prepare") == 0
    assert run("tile", "--shard", "0/2") == 0
    assert run("tile", "--shard", "1/2") == 0
    assert run("split") == 0
    assert run("extract-features", "-m", "rgb_stats") == 0
    assert run("collect-features", "-m", "rgb_stats") == 0
    assert run("train", "-m", "rgb_stats", "--marker", "H3K27M", "CD3", "--fold", "all", "full") == 0
    assert run("evaluate", "-m", "rgb_stats") == 0
    assert run("plot-core", "-m", "rgb_stats", "--marker", "H3K27M", "--core", "A-1") == 0
    assert run("status") == 0
    return Config.load(cfg_path)


def test_outputs(project):
    cfg = project
    p = cfg.paths
    meta = pd.read_csv(p.core_metadata("train"), index_col=0)
    assert len(meta) == 12 and meta.n_tiles.gt(0).all()
    # only DIPG / DMG spinal / DMG Thalamic count as DMG (not e.g. "DMG Medulla")
    for role in ("train", "test"):
        m = pd.read_csv(p.core_metadata(role), index_col=0)
        assert m.loc[m.diagnosis == "DIPG", "is_DMG"].all()
        assert not m.loc[m.diagnosis == "DMG Medulla", "is_DMG"].any()
        assert (m.diagnosis == "DMG Medulla").any()
    labels = pd.read_csv(p.merged_labels("train"), index_col=0)
    assert labels.n_cells.min() >= cfg.tiling["min_cells_per_tile"]
    assert (labels.x1 - labels.x0).eq(cfg.tile_size_px).all() and cfg.tile_size_px == 307

    folds = json.loads(p.splits.read_text())["folds"]
    for f in folds:  # patient-level split
        tr = set(meta.loc[f["train_cores"], "patient"])
        va = set(meta.loc[f["val_cores"], "patient"])
        assert not tr & va
    assert sorted(sum((f["val_cores"] for f in folds), [])) == sorted(meta.index)

    # features are aligned with their tiles: red tiles have high mean red value
    X = pd.read_pickle(p.merged_features("rgb_stats", "train"))
    assert list(X.index) == list(labels.index)
    red = X.loc[labels.H3K27M > 0, "f0"].mean()
    other = X.loc[labels.H3K27M == 0, "f0"].mean()
    assert red > other

    summary = pd.read_csv(p.results_dir("rgb_stats") / "metrics_summary.csv")
    h3 = summary[(summary.set == "test") & (summary.marker == "H3K27M")]
    assert h3.roc_auc_mean.iloc[0] > 0.9
    assert {"test_DMG", "test_other_primary_tumors", "val"} <= set(summary.set)
    assert (p.results_dir("rgb_stats") / "metrics_full_model.csv").exists()
    assert (p.results_dir("rgb_stats") / "spatial_maps" / "A-1_273_H3K27M.png").exists()


def test_rerun_skips(project, caplog):
    cfg_path = project.base_dir / "config.yaml"
    with caplog.at_level("INFO"):
        cli.main(["train", "-c", str(cfg_path), "-m", "rgb_stats", "--marker", "H3K27M", "--fold", "0"])
    assert "skipping" in caplog.text
