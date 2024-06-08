import logging

from pip._internal.operations import freeze


def setup_custom_logger(name: str = "benchmark"):
    logger = logging.getLogger(name)

    sh = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    sh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.setLevel(level=logging.DEBUG)

    return logger


def print_requirements():
    pkgs = freeze.freeze()
    for pkg in pkgs:
        logger.info(pkg)


logger = setup_custom_logger()
