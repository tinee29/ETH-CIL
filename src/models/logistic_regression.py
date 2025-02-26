import math
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from random import sample
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm

import utils
import globals
import dataset.ImageDataset as ImageDataset
import models.PatchCNN
import models.UNetBase
import models.Block


def logistic_regression(train_images, val_images, train_masks, val_masks):
    train_patches, train_labels = utils.image_to_patches(train_images, train_masks)
    val_patches, val_labels = utils.image_to_patches(val_images, val_masks)
    x_train = utils.extract_features(train_patches)
    x_val = utils.extract_features(val_patches)
    clf = LogisticRegression(class_weight="balanced").fit(x_train, train_labels)
    print(f"Training accuracy: {clf.score(x_train, train_labels)}")
    print(f"Validation accuracy: {clf.score(x_val, val_labels)}")
    test_path = os.path.join(globals.ROOT_PATH, "test", "images")
    test_filenames = sorted(glob(test_path + "/*.png"))
    test_images = utils.load_all_from_path(test_path)
    test_patches = utils.image_to_patches(test_images)
    x_test = utils.extract_features(test_patches)
    test_pred = clf.predict(x_test).reshape(
        -1, test_images.shape[1] // globals.PATCH_SIZE, test_images.shape[2] // globals.PATCH_SIZE
    )
    utils.create_submission(
        test_pred, test_filenames, test_pred, submission_filename="logreg_submission.csv"
    )
