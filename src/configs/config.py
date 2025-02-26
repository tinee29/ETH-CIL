# Configuration files for experiments

import os
import torch

SEED = 42

# Path to the data directory
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
RAW_TRAINING_DIR = os.path.join(RAW_DATA_DIR, "training")
RAW_TEST_DIR = os.path.join(RAW_DATA_DIR, "test")
RAW_TRAINING_GROUNDTRUTH_DIR = os.path.join(RAW_TRAINING_DIR, "groundtruth")
RAW_TRAINING_IMAGES_DIR = os.path.join(RAW_TRAINING_DIR, "images")
RAW_TEST_IMAGES_DIR = os.path.join(RAW_TEST_DIR, "images")
EXTENDED_DATA_DIR = os.path.join(DATA_DIR, "extended")
EXTENDED_TRAINING_DIR = os.path.join(EXTENDED_DATA_DIR, "training")
EXTENDED_TRAINING_GROUNDTRUTH_DIR = os.path.join(EXTENDED_TRAINING_DIR, "groundtruth")
EXTENDED_TRAINING_IMAGES_DIR = os.path.join(EXTENDED_TRAINING_DIR, "images")
EXTENDED_VALIDATION_DIR = os.path.join(EXTENDED_DATA_DIR, "validation")
EXTENDED_VALIDATION_GROUNDTRUTH_DIR = os.path.join(EXTENDED_VALIDATION_DIR, "groundtruth")
EXTENDED_VALIDATION_IMAGES_DIR = os.path.join(EXTENDED_VALIDATION_DIR, "images")


# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = (400, 400)
# select cuda or mps gpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
# SETTINGS
MODEL_RESULTS_BASEDIR = "results/models"
VERBOSE = True

if __name__ == "__main__":
    # check if the directories exist
    assert os.path.exists(RAW_TRAINING_DIR), f"{RAW_TRAINING_DIR} does not exist"
    assert os.path.exists(RAW_TEST_DIR), f"{RAW_TEST_DIR} does not exist"
    assert os.path.exists(
        RAW_TRAINING_GROUNDTRUTH_DIR
    ), f"{RAW_TRAINING_GROUNDTRUTH_DIR} does not exist"
    assert os.path.exists(RAW_TRAINING_IMAGES_DIR), f"{RAW_TRAINING_IMAGES_DIR} does not exist"
