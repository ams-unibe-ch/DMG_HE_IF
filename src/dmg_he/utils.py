"""Small shared helpers."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

log = logging.getLogger("dmg_he")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_shard(shard: str | None) -> tuple[int, int]:
    """Parse ``"i/n"`` (0-based task i of n tasks). ``None`` -> all items.

    The SLURM placeholders ``SLURM`` are accepted for convenience:
    ``--shard slurm`` uses SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT.
    """
    if shard is None:
        return 0, 1
    if shard.lower() == "slurm":
        i = int(os.environ["SLURM_ARRAY_TASK_ID"]) - int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))
        n = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        return i, n
    i, n = (int(v) for v in shard.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"invalid shard {shard!r}: need 0 <= i < n")
    return i, n


def take_shard(items: list, shard: str | None) -> list:
    i, n = parse_shard(shard)
    return items[i::n]


def write_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Write atomically so that interrupted jobs never leave half-written files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, **kwargs)
    os.replace(tmp, path)


def require(path: Path, step: str) -> Path:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found - run `dmg-he {step}` first")
    return Path(path)
