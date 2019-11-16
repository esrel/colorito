import logging


def setup_logger(name, level=logging.INFO):
    if name:
        name = ':' + name.lstrip(':')

    logger = logging.getLogger(
             f'colorito{name}')

    logging.basicConfig( )
    logger.setLevel(level)

    return logger
