"""
This module configures, trains, and finetunes segmentation models using various encoder-architecture combinations. 
It loads and prepares datasets, creates models, and manages training processes with logging and checkpointing. 
Pre-trained weights are used when available, and specific models are finetuned for enhanced performance.
"""



####################################### BASE CONFIGURATION #########################################
from loguru import logger
import torch
import utils

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
# SUBMISSION
submission_dir = "submissions"
submission_weights = "submission_weights"
submission_checkpoints = "submission_checkpoints"
os.makedirs(submission_dir, exist_ok=True)
os.makedirs(submission_weights, exist_ok=True)
os.makedirs(submission_checkpoints, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")
logger.info(f"Parameters used: Decoder channels: {decoder_channels}, Learning rate: {lr}, Batch size: {batch_size}, Encoder weights: {encoder_weights}, Encoder depth: {encoder_depth}, Classes: {classes}, In channels: {in_channels}, Patience: {patience}, Seed: {seed}, Number of epochs: {nr_epochs}, Activation: {activation}")

####################################### BASE CONFIGURATION END #####################################

nr_models = 6
model_configs = [("efficientnet-b4", "unet++"), ("resnet50", "unet"), ("resnet50", "deeplabv3+"), ("resnet34", "unet++"), ("resnet50", "unet++"), ("xception", "unet++" )]
model_configs = model_configs[:nr_models]

################################## LOAD DATA #######################################################
from glob import glob
import globals
import torch

from dataset.LazyImageDataset import train_test_split_paths
from models.UNetEncoder import load_dataset

# RAW DATASET

def get_dataloader(seed, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), batch_size=globals.BATCH_SIZE):
    raw_image_paths = (
        glob(os.path.join(globals.ROOT_PATH, "training", "images", "*.png"))
        + glob(os.path.join(globals.ROOT_PATH, "training", "images", "*.jpg"))
    )
    raw_mask_paths = (
        glob(os.path.join(globals.ROOT_PATH, "training", "groundtruth", "*.png"))
        + glob(os.path.join(globals.ROOT_PATH, "training", "groundtruth", "*.jpg"))
    )
    raw_image_paths = sorted(raw_image_paths)
    raw_mask_paths = sorted(raw_mask_paths)
    assert len(raw_image_paths) == len(raw_mask_paths), "Number of images and masks must be equal"
    assert raw_image_paths[-20][-8:-4] == raw_mask_paths[-20][-8:-4], "Image and mask must have the same name"
    train_raw_image_paths, train_raw_mask_paths, val_raw_image_paths, val_raw_mask_paths = train_test_split_paths(
        raw_image_paths, raw_mask_paths, test_size=0.1, seed=seed
    )
    # CURATED DATASET
    curated_image_paths = (
        glob(os.path.join(globals.EXTERNAL_PATH, "curated_100", "images", "*.png"))
        + glob(os.path.join(globals.EXTERNAL_PATH, "curated_100", "images", "*.jpg"))
    )
    curated_mask_paths = (
        glob(os.path.join(globals.EXTERNAL_PATH, "curated_100", "masks", "*.png"))
        + glob(os.path.join(globals.EXTERNAL_PATH, "curated_100", "masks", "*.jpg"))
    )
    curated_image_paths = sorted(curated_image_paths)
    curated_mask_paths = sorted(curated_mask_paths)
    assert len(curated_image_paths) == len(curated_mask_paths), "Number of images and masks must be equal"
    assert curated_image_paths[-20][-8:-4] == curated_mask_paths[-20][-8:-4], "Image and mask must have the same name"
    train_curated_image_paths, train_curated_mask_paths, valid_curated_image_paths, valid_curated_mask_paths = train_test_split_paths(
        curated_image_paths, curated_mask_paths, test_size=0.01, seed=seed
    )
    train_image_paths_combined = train_raw_image_paths + train_curated_image_paths
    train_mask_paths_combined = train_raw_mask_paths + train_curated_mask_paths
    val_image_paths_combined = val_raw_image_paths + valid_curated_image_paths
    val_mask_paths_combined = val_raw_mask_paths + valid_curated_mask_paths

    # CUT TO MULTIPLE OF BATCH SIZE
    train_image_paths_combined = train_image_paths_combined[:len(train_image_paths_combined) - len(train_image_paths_combined) % batch_size]
    train_mask_paths_combined = train_mask_paths_combined[:len(train_mask_paths_combined) - len(train_mask_paths_combined) % batch_size]
    val_image_paths_combined = val_image_paths_combined[:len(val_image_paths_combined) - len(val_image_paths_combined) % batch_size]
    val_mask_paths_combined = val_mask_paths_combined[:len(val_mask_paths_combined) - len(val_mask_paths_combined) % batch_size]


    # COMBINED LAZY DATASET
    train_dataset_combined, val_dataset_combined = load_dataset(
        device, train_image_paths_combined, train_mask_paths_combined, val_image_paths_combined, val_mask_paths_combined, use_patches=False, is_lazy=True
    )

    combined_train_dataloader = torch.utils.data.DataLoader(
        train_dataset_combined, batch_size=batch_size, shuffle=True
    )
    combined_validation_dataloader = torch.utils.data.DataLoader(
        val_dataset_combined, batch_size=batch_size, shuffle=False
    )
    return combined_train_dataloader, combined_validation_dataloader

def get_raw_dataloader(seed, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), batch_size=globals.BATCH_SIZE):
    raw_image_paths = (
        glob(os.path.join(globals.ROOT_PATH, "training", "images", "*.png"))
        + glob(os.path.join(globals.ROOT_PATH, "training", "images", "*.jpg"))
    )
    raw_mask_paths = (
        glob(os.path.join(globals.ROOT_PATH, "training", "groundtruth", "*.png"))
        + glob(os.path.join(globals.ROOT_PATH, "training", "groundtruth", "*.jpg"))
    )
    raw_image_paths = sorted(raw_image_paths)
    raw_mask_paths = sorted(raw_mask_paths)
    assert len(raw_image_paths) == len(raw_mask_paths), "Number of images and masks must be equal"
    assert raw_image_paths[-20][-8:-4] == raw_mask_paths[-20][-8:-4], "Image and mask must have the same name"
    train_raw_image_paths, train_raw_mask_paths, val_raw_image_paths, val_raw_mask_paths = train_test_split_paths(
        raw_image_paths, raw_mask_paths, test_size=0.1, seed=seed
    )
    
    # CUT TO MULTIPLE OF BATCH SIZE
    train_raw_image_paths = train_raw_image_paths[:len(train_raw_image_paths) - len(train_raw_image_paths) % batch_size]
    train_raw_mask_paths = train_raw_mask_paths[:len(train_raw_mask_paths) - len(train_raw_mask_paths) % batch_size]
    val_raw_image_paths = val_raw_image_paths[:len(val_raw_image_paths) - len(val_raw_image_paths) % batch_size]
    val_raw_mask_paths = val_raw_mask_paths[:len(val_raw_mask_paths) - len(val_raw_mask_paths) % batch_size]


    # COMBINED LAZY DATASET
    train_dataset_raw, val_dataset_raw = load_dataset(
        device, train_raw_image_paths, train_raw_mask_paths, val_raw_image_paths, val_raw_mask_paths, use_patches=False, is_lazy=True
    )

    raw_train_dataloader = torch.utils.data.DataLoader(
        train_dataset_raw, batch_size=batch_size, shuffle=True
    )
    raw_validation_dataloader = torch.utils.data.DataLoader(
        val_dataset_raw, batch_size=batch_size, shuffle=False
    )
    return raw_train_dataloader, raw_validation_dataloader
    ################################## LOAD DATA END ###################################################

import segmentation_models_pytorch as smp
from models.UNetEncoder import UNetEncoder

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

# CREATE MODEL FUNCTION

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


from time import sleep

for i, (encoder_name, architecture) in enumerate(model_configs):
    combined_train_dataloader, combined_validation_dataloader = get_dataloader(seed + i, device, batch_size)
    model = create_model(encoder_name, architecture)
    model = model.to(device)
    model_name = f"{encoder_name}_{architecture}"
    logger.info(f"Training model: {model_name}")
    # check if submission weights exist at submission_weights_path and starts with model_name
    submission_weights_path = os.listdir(submission_weights)
    submission_weights_path = [os.path.join(submission_weights, path) for path in submission_weights_path if path.startswith(model_name) and (not "++" in path or "unet++" in model_name)]   
    submission_weights_path = submission_weights_path[0] if submission_weights_path else None
    if submission_weights_path:
        logger.info(f"Loading model weights from {submission_weights_path} as it was already trained.")
        model.load_state_dict(torch.load(submission_weights_path))
    else:
        logger.info(f"No weights found at {submission_weights_path}. Training from scratch.")
        os.makedirs(submission_weights_path := "submission_checkpoints", exist_ok=True)
        for _ in range(2 if "resnet" in encoder_name else 1): # resnet models take longer to train (and are less likely to overfit due to freezing)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            utils.train(combined_train_dataloader, combined_validation_dataloader, model, loss_fn, metric_fns, optimizer, nr_epochs, patience, model_name, submission_weights_path)
        submission_weights_path = os.path.join(submission_weights, f"{model_name}.pt")
        torch.save(model.state_dict(), submission_weights_path)
        sleep(2)
        utils.create_test_submission(model, device=device, submission_filename=os.path.join(submission_dir, f"{model_name}.csv"))
    

from time import sleep

# Out UNet implementation seems to benefit from finetuning, while the other models do not
nr_epochs_finetune = 24
patience_finetune = 8
finetune_li = [("efficientnet-b4", "unet++"), ("resnet50", "unet"), ("resnet50", "deeplabv3+")] # needed to set correct seed
for i, (encoder_name, architecture) in enumerate(model_configs):
    if not (encoder_name, architecture) in finetune_li:
        logger.info(f"Skipping finetuning for {encoder_name}_{architecture}")
        continue
    raw_train_dataloader, raw_validation_dataloader = get_raw_dataloader(seed + i, device, batch_size)
    combined_train_dataloader, combined_validation_dataloader = get_dataloader(seed + i, device, batch_size)
    model = create_model(encoder_name, architecture)
    model = model.to(device)
    model_name = f"{encoder_name}_{architecture}"
    logger.info(f"Training model: {model_name}")
    # check if submission weights exist at submission_weights_path and starts with model_name
    submission_weights_path = os.listdir(submission_weights)
    submission_path_finetuned = [os.path.join(submission_weights, path) for path in submission_weights_path if path.startswith(model_name) and (not "++" in path or "unet++" in model_name) and "finetuned" in path]
    if submission_path_finetuned:
        submission_path_finetuned = submission_path_finetuned[0]
        logger.info(f"Loading model weights from {submission_path_finetuned} as it was already finetuned.")
        model.load_state_dict(torch.load(submission_path_finetuned))
        continue
    submission_weights_path = [os.path.join(submission_weights, path) for path in submission_weights_path if path.startswith(model_name) and (not "++" in path or "unet++" in model_name)]    
    submission_weights_path = submission_weights_path[0] if submission_weights_path else None
    if submission_weights_path:
        logger.info(f"Loading model weights from {submission_weights_path}. Starting finetuning.")
        model.load_state_dict(torch.load(submission_weights_path))
        model_name = f"{encoder_name}_{architecture}-finetuned"
        os.makedirs(submission_weights_path := "submission_checkpoints", exist_ok=True)
        for _ in range(2 if "resnet" in encoder_name else 1): # our unet takes longer to train
            optimizer = torch.optim.Adam(model.parameters(), lr=lr / 4) # mutable
            utils.train(raw_train_dataloader, combined_validation_dataloader, model, loss_fn, metric_fns, optimizer, nr_epochs_finetune, patience_finetune, model_name, submission_weights_path)
        submission_weights_path = os.path.join(submission_weights, f"{model_name}.pt")
        torch.save(model.state_dict(), submission_weights_path)
        sleep(2)
        utils.create_test_submission(model, device=device, submission_filename=os.path.join(submission_dir, f"{model_name}.csv"))
    else:
        logger.error(f"No weights found at {submission_weights_path}. Cannot finetune model.")
        