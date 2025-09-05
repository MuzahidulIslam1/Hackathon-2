#!/usr/bin/env python3
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import io
from pathlib import Path
from src.models.predict import predict_df, load_pipeline, load_encoder
from src.utils.logger import get_logger

logger = get_logger("flask_app")

app = Flask(__name__, template_folder="templates", static_folder="static")

MODEL = None
ENC = None

def get_models():
    """Load trained model + encoder once."""
    global MODEL, ENC
    try:
        if MODEL is None:
            MODEL = load_pipeline("models/trained_model.pkl")
        if ENC is None:
            ENC = load_encoder("models/label_encoder.pkl")
        return MODEL, ENC
    except Exception as e:
        logger.exception(f"Failed to load models: {e}")
        raise

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        model, enc = get_models()

        # ---- Case 1: CSV Upload ----
        if "file" in request.files and request.files["file"].filename:
            df = pd.read_csv(request.files["file"])
            out = predict_df(df, model=model, encoder=enc)

            # Create in-memory CSV
            output = io.StringIO()
            out.to_csv(output, index=False)
            output.seek(0)

            return send_file(
                io.BytesIO(output.getvalue().encode("utf-8")),
                mimetype="text/csv",
                as_attachment=True,
                download_name="predictions.csv"
            )

        # ---- Case 2: JSON / Form ----
        else:
            payload = request.get_json(silent=True) or request.form.to_dict(flat=True)
            for k, v in payload.items():
                try:
                    if isinstance(v, str) and v.isdigit():
                        payload[k] = int(v)
                except Exception:
                    pass
            df = pd.DataFrame([payload])
            out = predict_df(df, model=model, encoder=enc)
            return out.iloc[0].to_json()

    except Exception as e:
        logger.exception(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting Flask app...")
    app.run(host="0.0.0.0", port=5000, debug=False)
