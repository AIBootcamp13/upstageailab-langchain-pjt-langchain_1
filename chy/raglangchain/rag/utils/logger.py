import logging
import sys
from contextlib import contextmanager


def append_handler(logger):
    date_format = "%y%m%d-%H:%M:%S"
    fmt = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s", datefmt=date_format)

    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)

    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    return logger


def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger = append_handler(logger)
    return logger


@contextmanager
def log_block(logger):
    logger.info("=" * 80)
    try:
        yield
    finally:
        logger.info("=" * 80)
