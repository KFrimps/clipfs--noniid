from pathlib import Path

# Project root is two levels up from this file (src/utils/paths.py -> src/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Directory for raw/downloaded data
DATA_DIR = PROJECT_ROOT / "data"

# Directory for storing extracted CLIP feature tensors
FEATURES_DIR = DATA_DIR / "features"

# Directory for training run outputs (CSV logs, etc.)
RUNS_DIR = PROJECT_ROOT / "runs"
