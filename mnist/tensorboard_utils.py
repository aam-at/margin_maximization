from __future__ import absolute_import, division, print_function

from tensorboard import FileWriter
from tensorboard.summary import histogram, scalar

_tf_logger = None


def configure(logdir, flush_secs=2):
    """Configure logging: a file will be written to logdir, and flushed every
    flush_secs.

    """
    global _tf_logger
    if _tf_logger is not None:
        raise ValueError
    _tf_logger = FileWriter(logdir, flush_secs=flush_secs)


def log_value(name, value, step=None):
    if _tf_logger is None:
        return ValueError
    _tf_logger.add_summary(scalar(name, value), global_step=step)


def log_histogram(name, values, step=None):
    if _tf_logger is None:
        return ValueError
    _tf_logger.add_summary(histogram(name, values), global_step=step)
