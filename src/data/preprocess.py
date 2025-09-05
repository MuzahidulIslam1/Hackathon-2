
import pandas as pd
from pathlib import Path

def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    # drop completely empty or unnamed columns produced by certain CSV exports
    cols_to_drop = [c for c in df.columns if c.lower().startswith("unnamed") or df[c].isna().all()]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df

def load_data(train_path: str, test_path: str):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train = _drop_unnamed(train)
    test = _drop_unnamed(test)
    return train, test

def get_feature_target(train_df):
    # target in this dataset is 'prognosis'
    if "prognosis" not in train_df.columns:
        raise KeyError("Expected target column 'prognosis' not found in training data")
    X = train_df.drop(columns=["prognosis"])
    y = train_df["prognosis"]
    return X, y
