from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pathlib import Path
import pandas as pd
from src.data.preprocess import load_data, get_feature_target
from src.utils.helpers import save_model
from src.utils.logger import get_logger
from sklearn.preprocessing import LabelEncoder

logger = get_logger("train")

def get_models():
    """Return dictionary of candidate models."""
    return {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "NaiveBayes": GaussianNB(),
        "SVM": SVC(kernel="linear", probability=True, random_state=42),
        "LogReg": LogisticRegression(max_iter=1000, random_state=42)
    }

def train_and_save(train_csv: str, test_csv: str, model_out_path: str,
                   encoder_out_path: str, preds_out_path: str, do_grid: bool = False):
    # we won’t use do_grid anymore, just ignore it

    try:
        logger.info("Loading data...")
        train_df, test_df = load_data(train_csv, test_csv)
        X_train, y_train = get_feature_target(train_df)
        X_test = test_df.drop(columns=["prognosis"]) if "prognosis" in test_df.columns else test_df.copy()
        y_test = test_df["prognosis"] if "prognosis" in test_df.columns else None

        # Label encode target
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test) if y_test is not None else None

        results = {}
        best_model, best_score, best_name = None, 0, None

        for name, model in get_models().items():
            logger.info(f"Training {name}...")
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95)),
                ("clf", model)
            ])
            pipe.fit(X_train, y_train_enc)

            if y_test_enc is not None:
                preds = pipe.predict(X_test)
                acc = accuracy_score(y_test_enc, preds)
                results[name] = acc
                logger.info(f"{name} accuracy: {acc:.4f}")
                if acc > best_score:
                    best_score, best_model, best_name = acc, pipe, name
            else:
                results[name] = None
                if best_model is None:  # fallback
                    best_model, best_name = pipe, name

        # Save best model + encoder
        save_model(best_model, model_out_path)
        save_model(le, encoder_out_path)
        logger.info(f"Best model: {best_name} with accuracy {best_score:.4f}")
        logger.info(f"Saved model to {model_out_path} and encoder to {encoder_out_path}")

        # Predictions with best model
        preds = best_model.predict(X_test)
        pred_labels = le.inverse_transform(preds.astype(int))
        out_df = X_test.copy()
        out_df["prediction"] = pred_labels
        out_df.to_csv(preds_out_path, index=False)
        logger.info(f"Saved predictions to {preds_out_path}")

        # Save results summary
        pd.DataFrame.from_dict(results, orient="index", columns=["accuracy"]).to_csv("model_performance.csv")
        logger.info("Saved model performance report: model_performance.csv")

        return True
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return False
