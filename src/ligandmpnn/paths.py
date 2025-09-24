import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent
SRC_DIR = PACKAGE_DIR.parent
REPO_DIR = SRC_DIR.parent

MODEL_PARAMS_DIR = Path(
    os.environ.get("LIGANDMPNN_MODEL_PARAMS_DIR", PACKAGE_DIR / "model_params")
)
MODEL_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
