import logging
from pathlib import Path
from termcolor import colored


def setup_logger(log_path: str, logger_name: str = "medfm"):
    """Create a process-local logger with file + console handlers.

    Args:
        log_path: Full log file path (e.g., /path/to/train.log)
        logger_name: Logger namespace
    """
    log_file_path = Path(log_path)
    if log_file_path.exists() and log_file_path.is_dir():
        log_file_path = log_file_path / "train.log"
    elif log_file_path.suffix == "":
        log_file_path = log_file_path / "train.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = "[%(asctime)s %(name)s] (%(filename)s:%(lineno)d): %(levelname)s %(message)s"
    color_fmt = (
        colored("[%(asctime)s %(name)s]", "green")
        + colored(" (%(filename)s:%(lineno)d)", "yellow")
        + ": %(levelname)s %(message)s"
    )

    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt))

    class ColoredFormatter(logging.Formatter):
        def format(self, record):
            return color_fmt % record.__dict__

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColoredFormatter(fmt))

    # Configure the named logger (propagate=False to avoid duplication via root).
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Also attach the file handler to the root logger so that any logger using
    # getLogger(__name__) (e.g. train.py → "__main__", utils/hardware.py →
    # "utils.hardware") writes to the same log file via normal propagation.
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Remove any existing file handler pointing to the same path to avoid
    # duplicate entries on repeated setup_logger calls.
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file_path):
            root.removeHandler(h)
            h.close()
    root.addHandler(file_handler)

    return logger
