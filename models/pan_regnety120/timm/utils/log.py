""" Logging helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
import logging
import logging.handlers


class FormatterNoInfo(logging.Formatter):
    def __init__(self, fmt='%(levelname)s: %(message)s'):
        """
        Initialize the logger.

        Args:
            self: (todo): write your description
            fmt: (str): write your description
        """
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        """
        Format log record.

        Args:
            self: (todo): write your description
            record: (todo): write your description
        """
        if record.levelno == logging.INFO:
            return str(record.getMessage())
        return logging.Formatter.format(self, record)


def setup_default_logging(default_level=logging.INFO, log_path=''):
    """
    Setup default logger.

    Args:
        default_level: (str): write your description
        logging: (todo): write your description
        INFO: (todo): write your description
        log_path: (str): write your description
    """
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FormatterNoInfo())
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)
    if log_path:
        file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=(1024 ** 2 * 2), backupCount=3)
        file_formatter = logging.Formatter("%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s")
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)
