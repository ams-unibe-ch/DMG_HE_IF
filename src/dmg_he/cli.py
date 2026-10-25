"""Command line interface: ``dmg-he <command> --config my_config.yaml``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config, default_config_text
from .utils import log, setup_logging

PIPELINE = """
pipeline (run in this order):
  1  prepare            split the cell-by-cell tables into cores
  2  tile               cut cores into 100 um tiles, label tiles      [--shard]
  3  split              merge labels, define the 5 CV folds
  4  extract-features   embed tiles with a foundation model           [--shard]
  5  collect-features   merge the per-core feature files
  6  train              lightGBM per marker and fold                  [--shard]
  7  evaluate           summary tables and plots
     plot-core          spatial prediction map for one core
     status             show what has been computed
     run-all            steps 1-7 for one model in a single process
"""


def _add_common(p):
    p.add_argument("-c", "--config", default=None, help="YAML config file (default: built-in defaults)")
    p.add_argument("-v", "--verbose", action="store_true")


def _add_shard(p):
    p.add_argument("--shard", default=None,
                   help="process part i of n, as 'i/n' (0-based), or 'slurm' to use the SLURM array task")


def _add_model(p, multiple=False):
    from .features import MODELS
    from .features.extractors import REGISTRY

    if multiple:
        p.add_argument("-m", "--model", nargs="+", required=True, choices=sorted(REGISTRY),
                       help=f"feature extractor(s), e.g. {' '.join(MODELS)}")
    else:
        p.add_argument("-m", "--model", required=True, choices=sorted(REGISTRY), help="feature extractor")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dmg-he", description="Predict MxIF marker positivity from virtual H&E tiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=PIPELINE)
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("init-config", help="write a config file with all defaults")
    p.add_argument("path", nargs="?", default="config.yaml")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")

    p = sub.add_parser("prepare", help="step 1: split cell tables into per-core tables")
    _add_common(p)
    p.add_argument("--block", choices=["train", "test"], help="only this block")

    p = sub.add_parser("tile", help="step 2: tile cores and compute tile labels")
    _add_common(p), _add_shard(p)
    p.add_argument("--block", choices=["train", "test"], help="only this block")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-png", action="store_true", help="only compute labels, do not write tile images")

    p = sub.add_parser("split", help="step 3: merge tile labels and make CV folds")
    _add_common(p)

    p = sub.add_parser("extract-features", help="step 4: tile embeddings")
    _add_common(p), _add_model(p), _add_shard(p)
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("collect-features", help="step 5: merge per-core feature files")
    _add_common(p), _add_model(p)

    p = sub.add_parser("train", help="step 6: train and predict")
    _add_common(p), _add_model(p), _add_shard(p)
    p.add_argument("--marker", nargs="+", default=["all"], help="marker name(s) or 'all' (default)")
    p.add_argument("--fold", nargs="+", default=["all"],
                   help="fold number(s), 'all' (default: all CV folds) and/or 'full' (train on the whole block)")
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("evaluate", help="step 7: performance tables and plots")
    _add_common(p), _add_model(p, multiple=True)

    p = sub.add_parser("plot-core", help="spatial prediction map for one test-block core")
    _add_common(p), _add_model(p)
    p.add_argument("--marker", required=True)
    p.add_argument("--core", required=True, nargs="+", help="core name(s), e.g. M-22 or M-22_273")

    p = sub.add_parser("status", help="show progress of each step")
    _add_common(p)

    p = sub.add_parser("run-all", help="run all steps for one model in a single process")
    _add_common(p), _add_model(p)
    p.add_argument("--fold", nargs="+", default=["all"])
    return parser


def status(cfg: Config) -> None:
    import pandas as pd

    from .features.extractors import REGISTRY

    paths = cfg.paths
    print(f"work dir: {paths.root}")
    for role in cfg.roles:
        mf = paths.core_metadata(role)
        if not mf.exists():
            print(f"  [{role}] prepare: not run")
            continue
        cores = pd.read_csv(mf, index_col=0).index
        tiled = sum(paths.tile_labels(role, c).exists() for c in cores)
        print(f"  [{role}] prepare: {len(cores)} cores | tile: {tiled}/{len(cores)} cores")
    print(f"  split: {'done' if paths.splits.exists() else 'not run'}")
    feat_root = paths.root / "features"
    models_root = paths.root / "models"
    n_jobs = len(cfg.marker_names) * cfg["cv"]["n_folds"]
    for model in sorted(REGISTRY):
        if not (feat_root / model).exists() and not (models_root / model).exists():
            continue
        n_feat = sum(1 for _ in (feat_root / model).glob("*/*.csv"))
        merged = all(paths.merged_features(model, r).exists() for r in cfg.roles)
        n_done = sum(1 for _ in (models_root / model).glob("*/fold*/metrics.json"))
        print(f"  {model}: features {n_feat} core files, merged={'yes' if merged else 'no'}, "
              f"trained {n_done}/{n_jobs} marker x fold")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init-config":
        path = Path(args.path)
        if path.exists() and not args.force:
            print(f"{path} exists - use --force to overwrite", file=sys.stderr)
            return 1
        path.write_text(default_config_text(), encoding="utf-8")
        print(f"wrote {path} - edit data_dir / work_dir and the file names, then run `dmg-he prepare -c {path}`")
        return 0

    setup_logging(args.verbose)
    cfg = Config.load(args.config)
    roles = [args.block] if getattr(args, "block", None) else None

    if args.command == "prepare":
        from . import prepare
        prepare.run(cfg, roles)
    elif args.command == "tile":
        from . import tiling
        tiling.run(cfg, roles, shard=args.shard, overwrite=args.overwrite, save_png=not args.no_png)
    elif args.command == "split":
        from . import splits
        splits.run(cfg)
    elif args.command == "extract-features":
        from . import features
        features.extract(cfg, args.model, shard=args.shard, overwrite=args.overwrite)
    elif args.command == "collect-features":
        from . import features
        features.collect(cfg, args.model)
    elif args.command == "train":
        from . import train
        train.run(cfg, args.model, markers=args.marker, folds=args.fold, shard=args.shard,
                  overwrite=args.overwrite)
    elif args.command == "evaluate":
        from . import evaluate
        evaluate.run(cfg, args.model)
    elif args.command == "plot-core":
        from . import evaluate
        for core in args.core:
            evaluate.plot_core(cfg, args.model, args.marker, core)
    elif args.command == "status":
        status(cfg)
    elif args.command == "run-all":
        from . import evaluate, features, prepare, splits, tiling, train
        prepare.run(cfg)
        tiling.run(cfg)
        splits.run(cfg)
        features.extract(cfg, args.model)
        features.collect(cfg, args.model)
        train.run(cfg, args.model, folds=args.fold)
        evaluate.run(cfg, [args.model])
    log.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
