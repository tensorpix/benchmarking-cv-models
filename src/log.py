import logging


def setup_custom_logger(name: str = "benchmark"):
    logger = logging.getLogger(name)

    sh = logging.StreamHandler()
    fh = logging.FileHandler("benchmark.log")

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    formatter_file = logging.Formatter(
        "%(asctime)s - %(module)s - %(levelname)s - %(message)s"
    )

    fh.setFormatter(formatter_file)
    sh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.setLevel(level=logging.DEBUG)

    return logger
