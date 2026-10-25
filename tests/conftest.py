import sys
from pathlib import Path

try:
    import lightgbm  # noqa: F401
except ImportError:  # fall back to a scikit-learn based stand-in
    sys.path.insert(0, str(Path(__file__).parent / "shims"))
