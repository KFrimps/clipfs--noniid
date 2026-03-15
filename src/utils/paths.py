from pathlib import Path

# Project root is two levels up from this file (src/utils/paths.py -> src/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Directory for storing extracted CLIP feature tensors
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
