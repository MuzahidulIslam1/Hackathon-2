
#!/usr/bin/env python3
import argparse
from src.models.train import train_and_save
from src.utils.logger import get_logger

logger = get_logger("cli_train")

def main():
    p = argparse.ArgumentParser(description="Train model and save artifacts")
    p.add_argument("--train", type=str, default="data/raw/Training.csv")
    p.add_argument("--test", type=str, default="data/raw/Testing.csv")
    p.add_argument("--model_out", type=str, default="models/trained_model.pkl")
    p.add_argument("--enc_out", type=str, default="models/label_encoder.pkl")
    p.add_argument("--preds_out", type=str, default="predictions.csv")
    p.add_argument("--grid", action="store_true", help="Run GridSearch (may be slow)")
    args = p.parse_args()
    logger.info("Starting training via CLI...")
    ok = train_and_save(args.train, args.test, args.model_out, args.enc_out, args.preds_out, do_grid=args.grid)
    if not ok:
        logger.error("Training failed. Check logs for details.")
        raise SystemExit(1)
    logger.info("Training completed successfully. Artifacts saved.")

if __name__ == "__main__":
    main()
