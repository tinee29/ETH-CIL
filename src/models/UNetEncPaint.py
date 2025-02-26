from torch import nn
from dataset.LazyImageDataset import RoadSegmentationDataset
from dataset.transforms import tensor_transforms
from models.Block import Block
import torch
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import torch
import cv2
from torch import nn


from models.MaskInpaintModel import SimpleAutoencoder
from models.UNetEncoder import UNetEncoder, load_dataset
import utils
import globals
from dataset.ImageDataset import ImageDataset
from logger import LOG_DIR, logger
from loss import ReachabilityLoss, WeightedBCELoss


class UNetEncPaint(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.unet = UNetEncoder()
        self.paint = SimpleAutoencoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unet(x)
        # x = torch.round((x > globals.CUTOFF).float())
        x = self.paint(x)
        return x
    
    def freeze_encoder(self):
        self.unet.freeze_encoder()

    def load_unet_weights(self, path):
        self.unet.load_state_dict(torch.load(path))

    def load_paint_weights(self, path):
        self.paint.load_state_dict(torch.load(path))

    def freeze_unet(self):
        self.unet.full_freeze()

    def load_weights(self, path):
        logger.info(f"Loading weights from {path}")
        self.load_state_dict(torch.load(path))

from segmentation_models_pytorch.losses import DiceLoss

def run_unet_encoder_paint(train_images, train_masks, val_images, val_masks, is_lazy=False, tensor_transforms_default=tensor_transforms, model=None, nr_epochs=10, optimizer=None, loss_fn=None, checkpoint_dir="checkpoints"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # reshape the image to simplify the handling of skip connections and maxpooling
    train_dataset, val_dataset = load_dataset(device, train_images, train_masks, val_images, val_masks, use_patches=False, is_lazy=is_lazy, tensor_transforms_default=tensor_transforms_default)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=globals.BATCH_SIZE, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=globals.BATCH_SIZE, shuffle=True
    )
    if model is None:
        model = UNetEncPaint().to(device)
    else:
        model = model.to(device)
        model.train()

    model.freeze_unet()
    loss_fn = loss_fn or WeightedBCELoss(1.) # WeightedBCELoss(1.)
    logger.info(f"Using loss function: {loss_fn}")
    metric_fns = {
        "acc": utils.accuracy_fn,
        "patch_acc": utils.patch_accuracy_fn,
        "f1": utils.f1_score_fn,
        "patch_f1": utils.patch_f1_score_fn,
    }
    optimizer = optimizer or torch.optim.Adam(model.parameters())
    utils.train(train_dataloader, val_dataloader, model, loss_fn, metric_fns, optimizer, nr_epochs, model_name="unet_encoder_paint", patience=10, checkpoint_dir=checkpoint_dir)

    del train_images, train_masks, val_images, val_masks  # free up memory
    return model