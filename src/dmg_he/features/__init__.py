"""Step 3b/4 - tile embeddings with pretrained pathology foundation models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from ..config import Config
from ..utils import log, require, take_shard, write_csv
from .extractors import REGISTRY, build

MODELS = ["retccl", "ctranspath", "remedis", "uni", "virchow", "provgigapath"]


def _all_cores(cfg: Config):
    items = []
    for role in cfg.roles:
        meta = pd.read_csv(require(cfg.paths.core_metadata(role), "prepare"), index_col=0)
        items += [(role, c) for c in meta.index]
    return items


def extract(cfg: Config, model: str, shard: str | None = None, overwrite: bool = False) -> None:
    """Embed all tiles of all cores (both blocks); one CSV per core."""
    paths = cfg.paths
    items = take_shard(_all_cores(cfg), shard)
    todo = [(r, c) for r, c in items if overwrite or not paths.core_features(model, r, c).exists()]
    log.info("[%s] extracting features for %d of %d cores", model, len(todo), len(items))
    if not todo:
        return
    extractor = build(model, cfg["feature_extractors"], cfg.resolve)
    for i, (role, core) in enumerate(todo, 1):
        labels = pd.read_csv(require(paths.tile_labels(role, core), "tile"))
        png_dir = paths.tile_png_dir(role, core)
        if labels.empty:
            feats = pd.DataFrame(index=pd.Index([], name="tile_id"))
        else:
            images = []
            for t in labels.tile_id:
                with Image.open(require(png_dir / f"{t}.png", "tile")) as im:
                    images.append(im.convert("RGB"))
            emb = extractor(images)
            assert emb.shape[0] == len(images)
            feats = pd.DataFrame(emb.astype(np.float32), index=pd.Index(labels.tile_id, name="tile_id"),
                                 columns=[f"f{j}" for j in range(emb.shape[1])])
        write_csv(feats, paths.core_features(model, role, core))
        log.info("[%s] %d/%d %s: %d tiles x %d features", model, i, len(todo), core, *feats.shape)


def collect(cfg: Config, model: str) -> None:
    """Merge the per-core feature files of each block into one table."""
    paths = cfg.paths
    for role in cfg.roles:
        labels = pd.read_csv(require(paths.merged_labels(role), "split"), index_col=0)
        frames, missing = [], []
        for core in labels.core_id.unique():
            f = paths.core_features(model, role, core)
            if f.exists():
                frames.append(pd.read_csv(f, index_col=0))
            else:
                missing.append(core)
        if missing:
            raise FileNotFoundError(f"[{model}/{role}] features missing for {len(missing)} cores "
                                    f"(e.g. {missing[:3]}) - run `dmg-he extract-features --model {model}`")
        feats = pd.concat(frames)
        lost = labels.index.difference(feats.index)
        if len(lost):
            raise ValueError(f"[{model}/{role}] {len(lost)} tiles have no features (e.g. {list(lost[:3])})")
        feats = feats.loc[labels.index]
        out = paths.merged_features(model, role)
        out.parent.mkdir(parents=True, exist_ok=True)
        feats.to_pickle(out)
        log.info("[%s/%s] %d tiles x %d features -> %s", model, role, *feats.shape, out)


def load(cfg: Config, model: str, role: str) -> pd.DataFrame:
    return pd.read_pickle(require(cfg.paths.merged_features(model, role), f"collect-features --model {model}"))


__all__ = ["MODELS", "REGISTRY", "extract", "collect", "load"]
