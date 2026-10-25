"""Configuration loading and the file layout of the working directory."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

DEFAULT_CONFIG_NAME = "default_config.yaml"


def default_config_text() -> str:
    return resources.files("dmg_he").joinpath(DEFAULT_CONFIG_NAME).read_text(encoding="utf-8")


# sections that are replaced as a whole instead of merged key by key
_REPLACE = {"markers", "param_grid", "subgroups"}


def _deep_update(base: dict, new: dict) -> dict:
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict) and key not in _REPLACE:
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


@dataclass
class Config:
    """Parsed configuration. ``raw`` holds the merged YAML dictionary."""

    raw: dict
    base_dir: Path

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        raw = yaml.safe_load(default_config_text())
        base_dir = Path.cwd()
        if path is not None:
            path = Path(path).expanduser().resolve()
            with open(path, encoding="utf-8") as fh:
                user = yaml.safe_load(fh) or {}
            raw = _deep_update(copy.deepcopy(raw), user)
            base_dir = path.parent
        cfg = cls(raw=raw, base_dir=base_dir)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        t = self.tiling
        if t["tile_size_um"] <= 0 or not 0 < t["stride_fraction"] <= 1:
            raise ValueError("tiling.tile_size_um must be > 0 and 0 < tiling.stride_fraction <= 1")
        if set(self.raw["blocks"]) != {"train", "test"}:
            raise ValueError("'blocks' must define exactly a 'train' and a 'test' block")
        for name, spec in self.markers.items():
            if spec.get("compartment") not in ("Cell", "Nucleus", "Cytoplasm"):
                raise ValueError(f"marker {name}: compartment must be Cell, Nucleus or Cytoplasm")

    # ---------------------------------------------------------------- helpers
    def resolve(self, p: str | Path) -> Path:
        p = Path(p).expanduser()
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    def __getitem__(self, key):
        return self.raw[key]

    @property
    def data_dir(self) -> Path:
        return self.resolve(self.raw["data_dir"])

    @property
    def work_dir(self) -> Path:
        return self.resolve(self.raw["work_dir"])

    @property
    def markers(self) -> dict:
        return self.raw["markers"]

    @property
    def marker_names(self) -> list[str]:
        return list(self.raw["markers"])

    @property
    def tiling(self) -> dict:
        return self.raw["tiling"]

    def block(self, role: str) -> dict:
        """Block spec ('train' or 'test') with resolved input paths."""
        spec = dict(self.raw["blocks"][role])
        for key in ("cells", "thresholds", "slide"):
            spec[key] = self.data_dir / spec[key]
        spec["role"] = role
        return spec

    @property
    def roles(self) -> list[str]:
        return ["train", "test"]

    @property
    def metadata_file(self) -> Path:
        return self.data_dir / self.raw["metadata"]

    # tile geometry in pixels (same rounding as the original analysis)
    @property
    def tile_size_px(self) -> int:
        return int(self.tiling["tile_size_um"] / self.tiling["resolution_um_per_px"])

    @property
    def stride_px(self) -> int:
        t = self.tiling
        return int(t["stride_fraction"] * t["tile_size_um"] / t["resolution_um_per_px"])

    @property
    def paths(self) -> "WorkPaths":
        return WorkPaths(self.work_dir)


class WorkPaths:
    """Layout of the working directory::

        work/
          cores/<role>/metadata.csv            core metadata (one row per core)
          cores/<role>/cells/<core>.csv        per-core cell positions and intensities
          cores/<role>/images/<core>.png       core image cropped from the slide
          tiles/<role>/labels/<core>.csv       per-core tile coordinates and marker counts
          tiles/<role>/png/<core>/<tile>.png   tile images
          tiles/labels_<role>.csv              all tile labels of a block (written by `split`)
          splits.json                          cross-validation folds (written by `split`)
          features/<model>/<role>/<core>.csv   per-core tile embeddings
          features/<model>/features_<role>.pkl merged embeddings (written by `collect-features`)
          models/<model>/<marker>/<fold>/      trained model, predictions, metrics
          results/<model>/                     evaluation tables and plots
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    def cores_dir(self, role):
        return self.root / "cores" / role

    def core_metadata(self, role):
        return self.cores_dir(role) / "metadata.csv"

    def core_cells(self, role, core):
        return self.cores_dir(role) / "cells" / f"{core}.csv"

    def core_image(self, role, core):
        return self.cores_dir(role) / "images" / f"{core}.png"

    def tile_labels(self, role, core):
        return self.root / "tiles" / role / "labels" / f"{core}.csv"

    def tile_png_dir(self, role, core):
        return self.root / "tiles" / role / "png" / core

    def merged_labels(self, role):
        return self.root / "tiles" / f"labels_{role}.csv"

    @property
    def splits(self):
        return self.root / "splits.json"

    def core_features(self, model, role, core):
        return self.root / "features" / model / role / f"{core}.csv"

    def merged_features(self, model, role):
        return self.root / "features" / model / f"features_{role}.pkl"

    def model_dir(self, model, marker, fold):
        return self.root / "models" / model / marker / fold_name(fold)

    def results_dir(self, model):
        return self.root / "results" / model


def fold_name(fold) -> str:
    return "full" if str(fold) == "full" else f"fold{int(fold)}"
