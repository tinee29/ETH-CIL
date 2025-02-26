import cv2
import numpy as np
from loguru import logger
import torch
import utils

from glob import glob
import globals
import torch
import os

from dataset.LazyImageDataset import train_test_split_paths
from models.UNetEncoder import load_dataset

import segmentation_models_pytorch as smp
from models.UNetEncoder import UNetEncoder

import random
from logger import LOG_DIR
from models.MaskInpaintModel import SimpleAutoencoder
import matplotlib.pyplot as plt
from time import sleep

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import sys, os
os.chdir(os.path.dirname(sys.path[0]))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'src'))

decoder_channels = (256, 128, 64, 32, 16) 
lr = 0.001
batch_size = 6
encoder_weights = "imagenet" 
encoder_depth = len(decoder_channels)
classes = 1
in_channels = 3
patience = 11
seed = 42
nr_epochs = 80
activation = None

submission_dir = "submissions"
submission_weights = "submission_weights"
submission_checkpoints = "submission_checkpoints"
os.makedirs(submission_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

loss_fn = None
metric_fns_base = {
    "acc": utils.accuracy_fn,
    "patch_acc": utils.patch_accuracy_fn,
    "f1": utils.f1_score_fn,
    "patch_f1": utils.patch_f1_score_fn,
}
metric_fns_sig = {
    "acc": lambda y1, y2: utils.accuracy_fn(torch.sigmoid(y1), y2),
    "patch_acc": lambda y1, y2: utils.patch_accuracy_fn(torch.sigmoid(y1), y2),
    "f1": lambda y1, y2: utils.f1_score_fn(torch.sigmoid(y1), y2),
    "patch_f1": lambda y1, y2: utils.patch_f1_score_fn(torch.sigmoid(y1), y2),
}
metric_fns = None
freeze_encoder = False

def create_model(encoder_name, architecture):
    """Create the model based on the encoder_name and the architecture
    Args:
        encoder_name (str): the name of the encoder, e.g. resnet50, efficientnet-b4, xception
        architecture (str): the name of the architecture, e.g. unet, unet++, deeplabv3+
    """
    global freeze_encoder, metric_fns, metric_fns_sig, loss_fn

    metric_fns = metric_fns_sig
    freeze_encoder = "resnet" in encoder_name
    loss_fn = torch.nn.BCEWithLogitsLoss()
    if architecture == "unet":
        loss_fn = torch.nn.BCELoss() # BCEWithLogitsLoss would apply sigmoid a second time
        metric_fns = metric_fns_base
        model = UNetEncoder()
        model.freeze_encoder()
    elif architecture == "unet++":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_channels=decoder_channels,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
    elif architecture == "deeplabv3+":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
    ########### FREEZE ENCODER IF NECESSARY ################
    if freeze_encoder and architecture != "unet":
        for child in list(model.encoder.children())[:5]:
            for param in child.parameters():
                param.requires_grad = False
    return model

class StreetArtist(torch.nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.models = models # list of models
        self.nr_models = len(models)
        self.street_paint = SimpleAutoencoder(self.nr_models) # one input channel for each model
    
    def forward(self, x):
        model_outputs = [model(x) for model in self.models]
        model_outputs = torch.cat(model_outputs, dim=1)
        return self.street_paint(model_outputs)
    
    def freeze_models(self):
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

nr_models = 6 # number of models to use in the street artist
model_configs = [("efficientnet-b4", "unet++"), ("resnet50", "unet"), ("resnet50", "deeplabv3+"), ("resnet34", "unet++"), ("resnet50", "unet++"), ("xception", "unet++" )]
model_configs_artist = model_configs[:nr_models]

models = [create_model(encoder_name, architecture) for encoder_name, architecture in model_configs_artist]
# load weights
for model, (encoder_name, architecture) in zip(models, model_configs_artist):
    model_name = f"{encoder_name}_{architecture}"
    submission_weights_path = os.listdir(submission_weights)
    submission_weights_path = [os.path.join(submission_weights, path) for path in submission_weights_path if path.startswith(model_name) and (not "++" in path or "unet++" in model_name)]
    submission_weights_path = sorted(submission_weights_path, key=lambda x: "finetuned" in x, reverse=True)
    submission_weights_path = submission_weights_path[0] if submission_weights_path else None
    print(submission_weights_path)
    model.load_state_dict(torch.load(submission_weights_path, map_location=device))
    model.to(device)

nr_epochs_street_paint = 17
patience_street_paint = 3

model_name = "street_paint" + "__".join([f"{encoder_name}_{architecture}" for encoder_name, architecture in model_configs_artist])
model_name += f"_{nr_epochs_street_paint}_{patience_street_paint}"
# model_name += f"_19_4" # best weights are at : street_paintefficientnet-b4_unet++__resnet50_unet__resnet50_deeplabv3+__resnet34_unet++__resnet50_unet++__xception_unet++.pt_17_3_19_4.pt

street_paint = StreetArtist(*models)
street_paint = street_paint.to(device)
  
logger.info(f"Loading model weights from {submission_weights_path + "/" + model_name + ".pt"}.")
street_paint.load_state_dict(torch.load(submission_weights + "/" + model_name + ".pt", map_location=device))





street_paint.to(device)
[street_paint.models[i].to(device) for i in range(len(street_paint.models))]
[street_paint.models[i].eval() for i in range(len(street_paint.models))]
street_paint.eval()
device_name = "cuda" if torch.cuda.is_available() else "cpu"
utils.create_test_submission(street_paint, device=device_name, submission_filename=os.path.join(submission_dir, f"{model_name.replace(".pt", "")}.csv"))