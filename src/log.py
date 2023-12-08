import logging


def setup_custom_logger(name: str = "benchmark"):
    logger = logging.getLogger(name)

    sh = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    sh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.setLevel(level=logging.DEBUG)

    return logger
