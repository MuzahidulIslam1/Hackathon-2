
import logging
from pathlib import Path

def get_logger(name: str = "ml_app", log_file: str = "models/logs/app.log"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    # ensure log directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(ch_fmt)

    # file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(ch_fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
