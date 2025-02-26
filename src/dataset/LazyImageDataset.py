import os
from typing import Tuple
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split
import globals


def train_test_split_paths(image_paths, mask_paths, test_size=0.2, seed=42):
    """It is assumed that image_paths and mask_paths are in the same order."""
    # Ensure image_paths and mask_paths have the same length
    assert len(image_paths) == len(
        mask_paths
    ), "image_paths and mask_paths must have the same length"

    # Create an array of indices
    indices = np.arange(len(image_paths))

    # Split the indices into train and validation sets
    train_indices, val_indices = train_test_split(indices, test_size=test_size, random_state=seed)

    # Use the indices to select the corresponding paths
    train_image_paths = [image_paths[i] for i in train_indices]
    train_mask_paths = [mask_paths[i] for i in train_indices]
    val_image_paths = [image_paths[i] for i in val_indices]
    val_mask_paths = [mask_paths[i] for i in val_indices]

    return train_image_paths, train_mask_paths, val_image_paths, val_mask_paths


class RoadSegmentationDataset(Dataset):
    def __init__(
        self,
        is_train,
        device,
        train_images_paths,
        train_masks_paths,
        val_images_paths,
        val_masks_paths,
        use_patches=True,
        resize_to=globals.RESIZE_TO,
        transform=None,
        tensor_transform=None,
    ):
        self.is_train = is_train
        self.device = device
        self.use_patches = use_patches
        self.resize_to = resize_to
        self.transform = transform
        self.tensor_transform = tensor_transform
        self.image_paths = train_images_paths if self.is_train else val_images_paths
        self.mask_paths = train_masks_paths if self.is_train else val_masks_paths
        self.nr_samples = len(self.image_paths)
        assert self.nr_samples == len(self.mask_paths), "Number of images and masks must be equal"

    def _load_image(self, image_path):
        return np.array(Image.open(image_path).convert("RGB")).astype(np.float32) / 255.0

    def _load_mask(self, mask_path):
        return np.array(Image.open(mask_path).convert("L")).astype(np.float32) / 255.0

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[Image.Image, Image.Image]:
        image = self._load_image(self.image_paths[idx])
        mask = self._load_mask(self.mask_paths[idx])

        image = cv2.resize(image, dsize=self.resize_to)
        mask = cv2.resize(mask, dsize=self.resize_to)

        if self.transform and self.is_train:
            image, mask = self.transform(image, mask)
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        # to device 
        image = image.to(self.device)
        mask = mask.to(self.device)

        if self.tensor_transform and self.is_train:
            image, mask = self.tensor_transform(image, mask)

        return image, mask
    
