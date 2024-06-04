import logging
import os
import sys
from datetime import datetime

import torch

from config import CFG


def set_logging(file_name="main"):
    if not os.path.exists(CFG.LOG_DIR):
        os.makedirs(CFG.LOG_DIR)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"Luxury_{file_name}_{current_time}.log"
    log_file_path = os.path.join(CFG.LOG_DIR, log_file_name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file_path, mode="w"), logging.StreamHandler()],
    )

    # Environment Info
    logger = logging.getLogger(__name__)
    logger.info("Environment Info:")
    logger.info("------------------------------------------------------------")
    logger.info("Author: Myungbin")
    logger.info(f"sys.platform: {sys.platform}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU 0: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA_HOME: {torch.utils.cmake_prefix_path}")
        logger.info(f"NVCC: {torch.version.cuda}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"PyTorch compiling details: {torch.__config__.show()}")


def train_log(*args):
    logging.info("Parameters Info:")
    logging.info("------------------------------------------------------------")
    logging.info(f"Batch Size: {CFG.BATCH_SIZE}")
    logging.info(f"Epoch: {CFG.EPOCHS}")
    logging.info(f"Learning Rate: {CFG.LEARNING_RATE}")
    for arg in args:
        logging.info(f"{arg.__class__.__name__}: {arg}")
