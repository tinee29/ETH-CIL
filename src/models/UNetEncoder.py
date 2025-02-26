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


import utils
import globals
from dataset.ImageDataset import ImageDataset
from logger import LOG_DIR, logger
from loss import WeightedBCELoss


def conv_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.LeakyReLU(inplace=True),
    )
    return block


def upconv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.LeakyReLU(inplace=True),
    )  # (H, W, in_channels) -> (2H, 2W, out_channels)


import torchvision.models as models


def get_pretrained_encoder():
    # Load a pretrained ResNet50 as an example
    encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Adjust the first convolutional layer to accept 3 channels
    encoder.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return encoder


resnet50_encoder = get_pretrained_encoder()


class UNetEncoder(nn.Module):
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self, encoder=resnet50_encoder, out_channels=1):
        super(UNetEncoder, self).__init__()

        # Encoder (pretrained)
        self.encoder = encoder
        self.encoder_layers = list(encoder.children())

        # Use the encoder layers
        self.enc1 = nn.Sequential(*self.encoder_layers[:3])  # (H, W, 3) -> (H/2, W/2, 64)
        self.enc2 = nn.Sequential(*self.encoder_layers[3:5])  # (H/2, W/2, 64) -> (H/4, W/4, 256)
        self.enc3 = self.encoder_layers[5]  # (H/4, W/4, 256) -> (H/8, W/8, 512)
        self.enc4 = self.encoder_layers[6]  # (H/8, W/8, 512) -> (H/16, W/16, 1024)

        self.bottleneck = self.encoder_layers[7]  # (H/16, W/16, 1024) -> (H/32, W/32, 2048)

        self.latent_work = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=1),
            nn.LeakyReLU(inplace=True),
        )

        # Decoder
        self.dec5 = upconv_block(2048, 1024)
        self.dec4 = upconv_block(1024, 512)
        self.dec3 = upconv_block(512, 256)
        self.dec2 = upconv_block(256, 64)
        self.dec1 = upconv_block(64, 64)

        # Final layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder forward pass
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Bottleneck forward pass
        bottleneck = self.bottleneck(enc4)

        # Decoder forward pass
        dec5 = self.dec5(bottleneck)
        dec4 = self.dec4(dec5)
        dec3 = self.dec3(dec4 + enc3)
        dec2 = self.dec2(dec3 + enc2)
        dec1 = self.dec1(dec2 + enc1)

        # Final layer
        final = self.final_conv(dec1)
        return torch.sigmoid(final)

    def freeze_encoder(self):
        logger.info("Freezing encoder weights enc1, enc2")
        for param in self.enc1.parameters():
            param.requires_grad = False
        for param in self.enc2.parameters():
            param.requires_grad = False
        for param in self.encoder.conv1.parameters():
            param.requires_grad = True
        # for param in self.enc3.parameters():
        #     param.requires_grad = False
        # for param in self.enc4.parameters():
        #     param.requires_grad = False
        # assert self.enc1[0].weight.requires_grad is False, "Encoder is not frozen"

    def full_freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        # for param in self.final_conv.parameters():
        #     param.requires_grad = True
        # logger.info("Fully frozen model")

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        logger.info(f"Loaded weights from {path}")


def load_dataset(
    device, train_images, train_masks, val_images, val_masks, use_patches, is_lazy=False, tensor_transforms_default=None
) -> tuple[RoadSegmentationDataset | ImageDataset, RoadSegmentationDataset | ImageDataset]:
    if is_lazy:
        train_dataset = RoadSegmentationDataset(
            True,
            device,
            train_images,
            train_masks,
            val_images,
            val_masks,
            use_patches=False,
            resize_to=(384, 384),
            tensor_transform=tensor_transforms_default or tensor_transforms,
        )
        val_dataset = RoadSegmentationDataset(
            False,
            device,
            train_images,
            train_masks,
            val_images,
            val_masks,
            use_patches=False,
            resize_to=(384, 384),
            tensor_transform=tensor_transforms_default or tensor_transforms,
        )
    else:
        train_dataset = ImageDataset(
            True,
            device,
            train_images,
            train_masks,
            val_images,
            val_masks,
            use_patches=False,
            resize_to=(384, 384),
        )
        val_dataset = ImageDataset(
            False,
            device,
            train_images,
            train_masks,
            val_images,
            val_masks,
            use_patches=False,
            resize_to=(384, 384),
        )
    return train_dataset, val_dataset

from segmentation_models_pytorch.losses import DiceLoss

def run_unet_encoder(
    train_images,
    train_masks,
    val_images,
    val_masks,
    is_lazy=False,
    model=None,
    optimizer=None,
    nr_epochs=22,
    tensor_transforms_default=None,
    loss_fn= WeightedBCELoss(1.), 
    checkpoint_dir="checkpoints",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # reshape the image to simplify the handling of skip connections and maxpooling
    train_dataset, val_dataset = load_dataset(
        device, train_images, train_masks, val_images, val_masks, use_patches=False, is_lazy=is_lazy, tensor_transforms_default=tensor_transforms_default
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=globals.BATCH_SIZE, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=globals.BATCH_SIZE, shuffle=True
    )
    if model is None:
        model = UNetEncoder().to(device)
    else:
        model = model.to(device)
        # model.train()
    model.freeze_encoder()
    loss_fn =loss_fn 
    logger.info(f"Using loss function: {loss_fn}")
    metric_fns = {
        "acc": utils.accuracy_fn,
        "patch_acc": utils.patch_accuracy_fn,
        "f1": utils.f1_score_fn,
        "patch_f1": utils.patch_f1_score_fn,
    }
    optimizer = optimizer or torch.optim.Adam(model.parameters())
    utils.train(train_dataloader, val_dataloader, model, loss_fn, metric_fns, optimizer, nr_epochs, model_name="unet_encoder", patience=7, checkpoint_dir=checkpoint_dir)

    # SAVE MODEL
    if globals.SAVE_MODEL:
        torch.save(model.state_dict(), f"model_weights_unet_resnet.pt")
        logger.info(
            f"Saved checkpoint after nr_epochs={nr_epochs} to model_weights_unet_resnet.pt."
        )

    del train_images, train_masks, val_images, val_masks  # free up memory
    return model

def run_unet_encoder_loader(
    train_dataloader,
    val_dataloader,
    is_lazy=False,
    model=None,
    optimizer=None,
    nr_epochs=22,
    tensor_transforms_default=None,
    loss_fn= WeightedBCELoss(1.),
    checkpoint_dir="checkpoints",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # reshape the image to simplify the handling of skip connections and maxpooling
    if model is None:
        model = UNetEncoder().to(device)
    else:
        model = model.to(device)
    model.freeze_encoder()
    loss_fn =loss_fn 
    logger.info(f"Using loss function: {loss_fn}")
    metric_fns = {
        "acc": utils.accuracy_fn,
        "patch_acc": utils.patch_accuracy_fn,
        "f1": utils.f1_score_fn,
        "patch_f1": utils.patch_f1_score_fn,
    }
    optimizer = optimizer or torch.optim.Adam(model.parameters())
    utils.train(train_dataloader, val_dataloader, model, loss_fn, metric_fns, optimizer, nr_epochs, model_name="unet_encoder", patience=7, checkpoint_dir=checkpoint_dir)

    # SAVE MODEL
    if globals.SAVE_MODEL:
        torch.save(model.state_dict(), f"model_weights_unet_resnet.pt")
        logger.info(
            f"Saved checkpoint after nr_epochs={nr_epochs} to model_weights_unet_resnet.pt."
        )
    return model
