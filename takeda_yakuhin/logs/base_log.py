from pathlib import Path
from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG
import datetime

def create_logger(exp_version):
    now = datetime.datetime.now()
    log_file = ("logs/log_list/{}_{}.log".format(now,exp_version))

    # logger
    logger_ = getLogger(exp_version)
    logger_.setLevel(DEBUG)

    # formatter
    fmr = Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")

    # file handler
    fh = FileHandler(log_file)
    fh.setLevel(DEBUG)
    fh.setFormatter(fmr)

    # stream handler
    ch = StreamHandler()
    ch.setLevel(INFO)
    ch.setFormatter(fmr)

    logger_.addHandler(fh)
    logger_.addHandler(ch)


def get_logger(exp_version):
    return getLogger(exp_version)


"""
from base_log import create_logger, get_logger

VERSION = "xxxx" # 実験番号

if __name__ == "__main__":
    create_logger(VERSION)
    get_logger(VERSION).info("メッセージ")
"""