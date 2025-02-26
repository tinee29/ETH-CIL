from torch import nn
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
import dataset.ImageDataset as ImageDataset


class UNet(nn.Module):
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        enc_chs = chs  # number of channels in the encoder
        dec_chs = chs[::-1][:-1]  # number of channels in the decoder
        self.enc_blocks = nn.ModuleList(
            [Block(in_ch, out_ch) for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])]
        )  # encoder blocks
        self.pool = nn.MaxPool2d(2)  # pooling layer (can be reused as it will not be trained)
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
                for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])
            ]
        )  # deconvolution
        self.dec_blocks = nn.ModuleList(
            [Block(in_ch, out_ch) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])]
        )  # decoder blocks
        self.head = nn.Sequential(
            nn.Conv2d(dec_chs[-1], 1, 1), nn.Sigmoid()
        )  # 1x1 convolution for producing the output

    def forward(self, x):
        # encode
        enc_features = []
        for block in self.enc_blocks[:-1]:
            x = block(x)  # pass through the block
            enc_features.append(x)  # save features for skip connections
            x = self.pool(x)  # decrease resolution
        x = self.enc_blocks[-1](x)
        # decode
        for block, upconv, feature in zip(self.dec_blocks, self.upconvs, enc_features[::-1]):
            x = upconv(x)  # increase resolution
            x = torch.cat([x, feature], dim=1)  # concatenate skip features
            x = block(x)  # pass through the block
        return self.head(x)  # reduce to 1 channel


def run_unet(train_images, train_masks, val_images, val_masks):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # reshape the image to simplify the handling of skip connections and maxpooling
    train_dataset = ImageDataset.ImageDataset(
        "training",
        device,
        train_images,
        train_masks,
        val_images,
        val_masks,
        use_patches=False,
        resize_to=(384, 384),
    )
    val_dataset = ImageDataset.ImageDataset(
        "validation",
        device,
        train_images,
        train_masks,
        val_images,
        val_masks,
        use_patches=False,
        resize_to=(384, 384),
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)
    model = UNet().to(device)
    loss_fn = nn.BCELoss()
    metric_fns = {
        "acc": utils.accuracy_fn,
        "patch_acc": utils.patch_accuracy_fn,
        "f1": utils.f1_score_fn,
    }
    optimizer = torch.optim.Adam(model.parameters())
    n_epochs = 5
    utils.train(train_dataloader, val_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs)

    # predict on test set
    test_path = os.path.join(globals.ROOT_PATH, "test", "images")
    test_filenames = glob(test_path + "/*.png")
    test_images = utils.load_all_from_path(test_path)
    batch_size = test_images.shape[0]
    size = test_images.shape[1:3]
    # resize the test images
    test_images = np.stack([cv2.resize(img, dsize=(384, 384)) for img in test_images], 0)
    test_images = test_images[:, :, :, :3]
    test_images = utils.np_to_tensor(np.moveaxis(test_images, -1, 1), device)
    test_pred = [model(t).detach().cpu().numpy() for t in test_images.unsqueeze(1)]
    test_pred = np.concatenate(test_pred, 0)
    test_pred = np.moveaxis(test_pred, 1, -1)  # CHW to HWC
    test_pred = np.stack(
        [cv2.resize(img, dsize=size) for img in test_pred], 0
    )  # resize to original shape
    # compute labels
    test_pred = test_pred.reshape(
        (
            -1,
            size[0] // globals.PATCH_SIZE,
            globals.PATCH_SIZE,
            size[0] // globals.PATCH_SIZE,
            globals.PATCH_SIZE,
        )
    )
    test_pred = np.moveaxis(test_pred, 2, 3)
    test_pred = np.round(np.mean(test_pred, (-1, -2)) > globals.CUTOFF)
    utils.create_submission(
        test_pred, test_filenames, test_pred, submission_filename="unet_submission.csv"
    )
