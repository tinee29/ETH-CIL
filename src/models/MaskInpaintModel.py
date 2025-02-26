import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple
import matplotlib.pyplot as plt

class SimpleAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 1) -> None:
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 7, stride=2, padding=3, dilation=2), # B, 16, 200, 200
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=3, dilation=2),  # B, 32, 100, 100
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),  # B, 64, 50, 50
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),  # B, 128, 25, 25
            nn.LeakyReLU(),
            # keep size 25x25
            nn.Conv2d(128, 128, 5, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 5, padding=2),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),  # B, 64, 50, 50
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1),  # B, 32, 100, 100
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 5, stride=2, padding=2, output_padding=1),  # B, 16, 200, 200
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 1, 7, stride=2, padding=3, output_padding=1),  # B, 1, 400, 400
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
