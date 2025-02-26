import os, sys, time
from globals import VERBOSE
import cv2
from loguru import logger

LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {module} | {message}"
ts = time.strftime("%Y-%m-%d__%H-%M-%S")
LOG_DIR = os.path.join("logs", "training", ts)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"log.log")
print(LOG_FILE)
logger.add(f"{LOG_FILE}", format=LOG_FORMAT, level="DEBUG" if VERBOSE else "INFO")
if VERBOSE:
    logger.add(sys.stdout, format=LOG_FORMAT, level="DEBUG")
