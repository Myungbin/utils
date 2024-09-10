import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
current_time = datetime.now().strftime("%Y%m%d")


class DefaultFormatter(logging.Formatter):
    def format(self, record):
        if record.funcName == "<module>":
            self._style._fmt = "[%(asctime)s][%(levelname)s][%(module)s]: %(message)s"
        else:
            self._style._fmt = "[%(asctime)s][%(levelname)s][%(module)s - %(funcName)s]: %(message)s"
        self.datefmt = "%H:%M:%S"
        return super().format(record)


default_formatter = DefaultFormatter()


def default_logger(logger):
    log_dir = os.path.join("log", current_time)
    os.makedirs(log_dir, exist_ok=True)
    file_debug_handler = logging.FileHandler(os.path.join(log_dir, f"default.log"))
    file_debug_handler.setLevel(logging.DEBUG)
    file_debug_handler.setFormatter(default_formatter)
    logger.addHandler(file_debug_handler)


def warning_logger(logger):
    log_dir = os.path.join("log", current_time)
    os.makedirs(log_dir, exist_ok=True)
    file_warning_handler = logging.FileHandler(os.path.join(log_dir, f"warning.log"))
    file_warning_handler.setLevel(logging.WARNING)
    file_warning_handler.setFormatter(default_formatter)
    logger.addHandler(file_warning_handler)


def console_logger(logger):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(default_formatter)
    logger.addHandler(console_handler)
