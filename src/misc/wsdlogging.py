import logging
from logging import StreamHandler


def get_info_logger(name, out_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    sh = StreamHandler()
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    if out_file is not None:
        fh = logging.FileHandler(out_file)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    return logger