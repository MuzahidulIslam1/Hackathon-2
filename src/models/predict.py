
from src.utils.helpers import load_model
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("predict")

MODEL_PATH = Path("models/trained_model.pkl")
ENC_PATH = Path("models/label_encoder.pkl")

def load_pipeline(model_path: str = None):
    try:
        p = model_path or str(MODEL_PATH)
        return load_model(p)
    except Exception as e:
        logger.exception(f"Failed to load pipeline: {e}")
        raise

def load_encoder(enc_path: str = None):
    try:
        p = enc_path or str(ENC_PATH)
        return load_model(p)
    except Exception as e:
        logger.exception(f"Failed to load encoder: {e}")
        raise

def predict_df(df: pd.DataFrame, model=None, encoder=None):
    try:
        pipe = model or load_pipeline()
        enc = encoder or load_encoder()
        preds = pipe.predict(df)
        labels = enc.inverse_transform(preds.astype(int))
        out = df.copy()
        out["prediction"] = labels
        return out
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise

def predict_one(record: dict, model=None, encoder=None):
    df = pd.DataFrame([record])
    return predict_df(df, model=model, encoder=encoder)
