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

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Reconfigure handlers on repeated setup to avoid duplicated logs.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

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

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
