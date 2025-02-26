import os
from typing import Tuple
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from random import randint
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import globals
from configs.config import DEVICE, SEED

resize_to = globals.RESIZE_TO


train_transform = lambda scale: A.Compose(
    [
        A.Resize(height=resize_to[0], width=resize_to[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomResizedCrop(
            height=resize_to[0], width=resize_to[1], scale=scale, ratio=(1, 1.2), p=1
        ),
        A.RandomBrightnessContrast(p=1, brightness_limit=0.2, contrast_limit=0.2),
        A.RandomShadow(p=0.5),
        A.RandomSnow(p=0.5),
        A.RandomRain(p=0.5),
        # hole dropout
        # A.CoarseDropout(p=0.5),
        ToTensorV2(),
    ]
)

train_transform = lambda scale: A.Compose(
    [
        A.Resize(height=resize_to[0], width=resize_to[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomResizedCrop(height=resize_to[0], width=resize_to[1], ratio=(1, 1.2), p=1),
        A.OneOf(
            [
                A.CoarseDropout(
                    max_holes=4,
                    max_height=40,
                    max_width=40,
                    min_holes=1,
                    min_height=10,
                    min_width=10,
                    fill_value=0,
                    p=1,
                ),
            ],
            p=0.25,
        ),
    ]
)

val_transforms = A.Compose(
    [
        A.Resize(height=resize_to[0], width=resize_to[1]),
        # ToTensorV2()
    ]
)

random_brightness = A.RandomBrightnessContrast(p=0.5, brightness_limit=0.1, contrast_limit=0.1)


class RoadSegmentationDataset(Dataset):
    def __init__(
        self,
        root_dir,
        max_samples=None,
        is_train=True,
        device=DEVICE,
        seed=SEED,
        split=1.0,
        **kwargs
    ):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.gt_dir = os.path.join(root_dir, "groundtruth")
        self.image_files = sorted(
            [f for f in os.listdir(self.image_dir) if f.endswith(".png") or f.endswith(".jpg")]
        )
        self.groundtruth_files = sorted([f for f in os.listdir(self.gt_dir) if f.endswith(".png")])
        self.transform = (
            train_transform(kwargs.get("scale", (0.8, 1))) if is_train else val_transforms
        )
        self.device = device
        if split < 1.0:
            n = int(split * len(self.image_files))
            self.random_indices = np.random.RandomState(seed=seed).permutation(
                len(self.image_files)
            )[0:n]
            if not is_train:
                self.random_indices = [
                    i for i in range(len(self.image_files)) if i not in self.random_indices
                ]
            self.image_files = [self.image_files[i] for i in self.random_indices]
            self.groundtruth_files = [self.groundtruth_files[i] for i in self.random_indices]

        if max_samples:
            r = randint(0, len(self.image_files) - max_samples)
            self.image_files = self.image_files[r : r + max_samples]
            self.groundtruth_files = self.groundtruth_files[r : r + max_samples]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        gt_name = os.path.join(self.gt_dir, self.groundtruth_files[idx])
        image = Image.open(img_name).convert("RGB")  # RGBA is default
        gt = Image.open(gt_name).convert("L")  # L is for grayscale
        # convert image to numpy array
        image = image = np.array(image).astype(np.float32) / 255.0  # TODO: check if this is correct
        gt = np.array(gt).astype(np.float32) / 255.0
        if self.transform:
            aug = self.transform(image=image, mask=gt)
            image = aug["image"]
            gt = aug["mask"]
            to_tensor = ToTensorV2()(image=image, mask=gt)
            image = to_tensor["image"]
            gt = to_tensor["mask"]

        image = image.to(self.device)
        gt = gt.to(self.device)
        gt = gt.unsqueeze(0).float()
        return image, gt


class RoadSegmentationDatasetFiles(Dataset):
    def __init__(
        self,
        image_files,
        gt_files,
        val_image_files,
        val_gt_files,
        max_samples=None,
        is_train=True,
        device=DEVICE,
        seed=SEED,
        split=1.0,
        **kwargs
    ):
        self.image_files = image_files if is_train else val_image_files
        self.groundtruth_files = gt_files if is_train else val_gt_files
        self.transform = (
            train_transform(kwargs.get("scale", (0.8, 1))) if is_train else val_transforms
        )
        self.device = device

        if max_samples:
            r = randint(0, len(self.image_files) - max_samples)
            self.image_files = self.image_files[r : r + max_samples]
            self.groundtruth_files = self.groundtruth_files[r : r + max_samples]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.image_files[idx]
        gt_name = self.groundtruth_files[idx]
        image = Image.open(img_name).convert("RGB")  # RGBA is default
        gt = Image.open(gt_name).convert("L")  # L is for grayscale
        # convert image to numpy array
        image = image = np.array(image).astype(np.float32) / 255.0  # TODO: check if this is correct
        gt = np.array(gt).astype(np.float32) / 255.0
        if self.transform:
            aug = self.transform(image=image, mask=gt)
            image = aug["image"]
            gt = aug["mask"]
            to_tensor = ToTensorV2()(image=image, mask=gt)
            image = to_tensor["image"]
            gt = to_tensor["mask"]

        image = image.to(self.device)
        gt = gt.to(self.device)
        gt = gt.unsqueeze(0).float()
        return image, gt


if __name__ == "__main__":
    from configs.config import RAW_TRAINING_DIR

    dataset = RoadSegmentationDataset(RAW_TRAINING_DIR, max_samples=10, is_train=True)
    print(len(dataset))
    image, gt = dataset[0]
    print(image.shape, gt.shape)
    print(image.min(), image.max(), gt.min(), gt.max())
    print(image.dtype, gt.dtype)
    print(image.device, gt.device)
