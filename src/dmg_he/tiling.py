"""Step 2 - cut every core into overlapping square tiles and label them.

A square window of ``tile_size_um`` is shifted over the bounding box of the
segmented cells of a core, first horizontally then vertically, by
``stride_fraction`` of the edge length. Windows containing at least
``min_cells_per_tile`` cells are kept. For each marker the number of cells whose
mean intensity exceeds the per-core threshold is stored; a tile counts as
positive if it holds at least ``min_positive_cells`` such cells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from .config import Config
from .utils import log, require, take_shard, write_csv


class SlideReader:
    """Read rectangular regions of a (large) RGB TIFF without loading it twice."""

    def __init__(self, path):
        import tifffile

        self.path = path
        self._arr = None
        try:  # uncompressed, contiguous TIFFs can be memory-mapped
            self._arr = tifffile.memmap(path, mode="r")
        except Exception:
            try:  # lazy, chunked access (requires zarr)
                import zarr

                store = tifffile.imread(path, aszarr=True)
                z = zarr.open(store, mode="r")
                self._arr = z[0] if hasattr(z, "keys") and "0" in z else z
            except Exception:
                log.info("loading full slide into memory: %s", path)
                self._arr = tifffile.imread(path)
        if self._arr.ndim != 3:
            raise ValueError(f"{path}: expected an RGB(A) image, got shape {self._arr.shape}")

    def region(self, x0, y0, x1, y1) -> np.ndarray:
        img = np.asarray(self._arr[y0:y1, x0:x1, :3])
        if img.dtype != np.uint8:
            raise TypeError(f"{self.path}: expected an 8-bit RGB slide, got dtype {img.dtype}")
        return img


def tile_windows(x, y, side, stride, min_cells):
    """Yield (x0, y0, mask) for every window with at least ``min_cells`` cells.

    ``x``/``y`` are cell centroids in pixels. Windows start at the minimum cell
    coordinate and stop before exceeding the maximum cell coordinate; cells on
    the window border are included.
    """
    x_min, x_max, y_min, y_max = x.min(), x.max(), y.min(), y.max()
    y0 = y_min
    while y0 + side < y_max:
        x0 = x_min
        in_row = (y >= y0) & (y <= y0 + side)
        while x0 + side < x_max:
            mask = in_row & (x >= x0) & (x <= x0 + side)
            if mask.sum() >= min_cells:
                yield int(x0), int(y0), mask
            x0 += stride
        y0 += stride


def tile_core(cfg: Config, role: str, core_id: str, meta_row: pd.Series, slide: SlideReader | None,
              save_png: bool = True) -> pd.DataFrame:
    paths = cfg.paths
    cells = pd.read_csv(paths.core_cells(role, core_id))
    side, stride = cfg.tile_size_px, cfg.stride_px
    t = cfg.tiling
    markers = [m for m in cfg.marker_names if m in cells.columns]
    thr = {m: meta_row.get(f"threshold_{m}", np.nan) for m in markers}
    x, y = cells.x_px.to_numpy(), cells.y_px.to_numpy()
    positive = {m: (cells[m].to_numpy() > thr[m]) for m in markers if pd.notna(thr[m])}

    rows = []
    png_dir = paths.tile_png_dir(role, core_id)
    for k, (x0, y0, mask) in enumerate(tile_windows(x, y, side, stride, t["min_cells_per_tile"])):
        tile_id = f"tile_{core_id}_{k}"
        row = {"tile_id": tile_id, "core_id": core_id, "x0": x0, "y0": y0,
               "x1": x0 + side, "y1": y0 + side, "n_cells": int(mask.sum())}
        for m in markers:
            row[m] = int(positive[m][mask].sum()) if m in positive else np.nan
        rows.append(row)
        if save_png and slide is not None:
            png_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(slide.region(x0, y0, x0 + side, y0 + side)).save(png_dir / f"{tile_id}.png")

    if slide is not None:
        x0, y0, x1, y1 = (int(meta_row[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
        img_path = paths.core_image(role, core_id)
        img_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(slide.region(x0, y0, x1, y1)).save(img_path)

    df = pd.DataFrame(rows, columns=["tile_id", "core_id", "x0", "y0", "x1", "y1", "n_cells", *markers])
    write_csv(df, paths.tile_labels(role, core_id), index=False)
    return df


def run(cfg: Config, roles=None, shard: str | None = None, overwrite: bool = False,
        save_png: bool = True) -> None:
    paths = cfg.paths
    for role in roles or cfg.roles:
        meta = pd.read_csv(require(paths.core_metadata(role), "prepare"), index_col=0)
        cores = take_shard(meta.index.to_list(), shard)
        todo = [c for c in cores if overwrite or not paths.tile_labels(role, c).exists()]
        log.info("[%s] tiling %d of %d cores (%d already done)", role, len(todo), len(cores), len(cores) - len(todo))
        if not todo:
            continue
        slide = SlideReader(cfg.block(role)["slide"]) if save_png else None
        for i, core in enumerate(todo, 1):
            df = tile_core(cfg, role, core, meta.loc[core], slide, save_png=save_png)
            log.info("[%s] %d/%d %s: %d tiles", role, i, len(todo), core, len(df))
