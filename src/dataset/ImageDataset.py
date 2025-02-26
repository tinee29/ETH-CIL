import math
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
from glob import glob
from random import sample, random
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import torchvision.transforms.functional as F
from torchvision import transforms


class ImageDataset(
    torch.utils.data.Dataset,
):
    # dataset class that deals with loading the data and making it available by index.

    def __init__(
        self,
        is_train,
        device,
        train_images,
        train_masks,
        val_images,
        val_masks,
        use_patches=True,
        resize_to=(400, 400),
    ):
        self.is_train = is_train
        self.device = device
        self.use_patches = use_patches
        self.resize_to = resize_to
        self.x, self.y, self.n_samples = None, None, None
        self.train_images = train_images
        self.train_masks = train_masks
        self.val_images = val_images
        self.val_masks = val_masks
        self._load_data()

    def _load_data(self):  # not very scalable, but good enough for now
        self.x = self.train_images if self.is_train else self.val_images
        self.y = self.train_masks if self.is_train else self.val_masks
        if self.use_patches:  # split each image into patches
            self.x, self.y = utils.image_to_patches(self.x, self.y)
        elif self.resize_to != (self.x.shape[1], self.x.shape[2]):  # resize images
            self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x], 0)
            self.y = np.stack([cv2.resize(mask, dsize=self.resize_to) for mask in self.y], 0)
        self.x = np.moveaxis(self.x, -1, 1)  # pytorch works with CHW format instead of HWC
        self.n_samples = len(self.x)

    def _preprocess(self, x: torch.Tensor, y: torch.Tensor):
        # to keep things simple we will not apply transformations to each sample,
        # but it would be a very good idea to look into preprocessing
        # horizontal flip, vertical flip, squish, crop, ...
        # TODO
        if self.is_train:
            if random() < 0.5:
                x = F.hflip(x)
                y = F.hflip(y)
            if random() < 0.5:
                x = F.vflip(x)
                y = F.vflip(y)
            # random zoom
            # i, j, h, w = transforms.RandomResizedCrop.get_params(
            #     x, scale=(0.8, 1.0), ratio=(1.0, 1.0)
            # )
            # x = F.resized_crop(x, i, j, h, w, size=(x.shape[1], x.shape[2]))
            # y = F.resized_crop(y, i, j, h, w, size=(y.shape[1], y.shape[2]))
            # random brightness, contrast, saturation
            # x = F.adjust_brightness(x, random() * 0.2 + 1)
            # x = F.adjust_contrast(x, random())
            # x = F.adjust_saturation(x, random())
        return x, y

    def __getitem__(self, item):
        return self._preprocess(
            utils.np_to_tensor(self.x[item], self.device),
            utils.np_to_tensor(self.y[[item]], self.device),
        )

    def __len__(self):
        return self.n_samples
