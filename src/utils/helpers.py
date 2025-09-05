
import joblib
from pathlib import Path
from .logger import get_logger

logger = get_logger()

def save_model(obj, path: str):
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, p)
        logger.info(f"Saved model to {p}")
    except Exception as e:
        logger.exception(f"Failed to save model to {path}: {e}")
        raise

def load_model(path: str):
    try:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model path not found: {p}")
        obj = joblib.load(p)
        logger.info(f"Loaded model from {p}")
        return obj
    except Exception as e:
        logger.exception(f"Failed to load model from {path}: {e}")
        raise
