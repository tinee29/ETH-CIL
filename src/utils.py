import math
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from configs.config import DEVICE
from early_stopping import EarlyStopping
import utils
from glob import glob
from random import sample
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import tqdm
from logger import logger, LOG_DIR

import globals

################# REPRODUCABILITY #################
import random


def set_seed(seed):
    logger.debug(f"Setting seed {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set the seed
seed = 42
set_seed(seed)
####################################################


def load_all_from_path(path, convert=lambda x: x, max_images=None, extra_paths=[]):
    # loads all HxW .pngs or .jpgs contained in path as a 4D np.array of shape (n_images, H, W, 3)
    # images are loaded as floats with values in the interval [0., 1.]
    return (
        np.stack(
            [
                np.array(convert(Image.open(f)))
                for f in load_all_image_paths(path, max_images, extra_paths)
            ]
        ).astype(np.float32)
        / 255.0
    )


def load_all_image_paths(path, max_images=None, extra_paths=[]):
    # returns a list of paths to all .pngs and .jpgs in path

    paths = sorted(sorted(glob(path + "/*.png")) + sorted(glob(path + "/*.jpg")))[:max_images]
    if max_images is not None and len(paths) < max_images:
        for p in extra_paths:
            paths += sorted(sorted(glob(p + "/*.png")) + sorted(glob(p + "/*.jpg")))[
                : max_images - len(paths)
            ]
    return paths


def show_first_n(imgs, masks, n=5):
    # visualizes the first n elements of a series of images and segmentation masks
    imgs_to_draw = min(5, len(imgs))
    fig, axs = plt.subplots(2, imgs_to_draw, figsize=(18.5, 6))
    for i in range(imgs_to_draw):
        axs[0, i].imshow(imgs[i])
        axs[1, i].imshow(masks[i])
        axs[0, i].set_title(f"Image {i}")
        axs[1, i].set_title(f"Mask {i}")
        axs[0, i].set_axis_off()
        axs[1, i].set_axis_off()
    plt.show()


def images_to_patches(images, masks=None):
    # takes in a 4D np.array containing images and (optionally) a 4D np.array containing the segmentation masks
    # returns a 4D np.array with an ordered sequence of patches extracted from the image and (optionally) a np.array containing labels
    n_images = images.shape[0]  # number of images
    h, w = images.shape[1:3]  # shape of images
    assert (h % globals.PATCH_SIZE) + (
        w % globals.PATCH_SIZE
    ) == 0  # make sure images can be patched exactly

    images = images[:, :, :, :3]

    h_patches = h // globals.PATCH_SIZE
    w_patches = w // globals.PATCH_SIZE

    patches = images.reshape(
        (n_images, h_patches, globals.PATCH_SIZE, w_patches, globals.PATCH_SIZE, -1)
    )
    patches = np.moveaxis(patches, 2, 3)
    patches = patches.reshape(-1, globals.PATCH_SIZE, globals.PATCH_SIZE, 3)
    if masks is None:
        return patches

    masks = masks.reshape(
        (n_images, h_patches, globals.PATCH_SIZE, w_patches, globals.PATCH_SIZE, -1)
    )
    masks = np.moveaxis(masks, 2, 3)
    labels = np.mean(masks, (-1, -2, -3)) > globals.CUTOFF  # compute labels
    labels = labels.reshape(-1).astype(np.float32)
    return patches, labels


def show_patched_image(patches, labels, h_patches=25, w_patches=25):
    # reorders a set of patches in their original 2D shape and visualizes them
    fig, axs = plt.subplots(h_patches, w_patches, figsize=(18.5, 18.5))
    for i, (p, l) in enumerate(zip(patches, labels)):
        # the np.maximum operation paints patches labeled as road red
        axs[i // w_patches, i % w_patches].imshow(np.maximum(p, np.array([l.item(), 0.0, 0.0])))
        axs[i // w_patches, i % w_patches].set_axis_off()
    plt.show()


def extract_features(x):
    return np.concatenate([np.mean(x, (-2, -3)), np.var(x, (-2, -3))], axis=-1)


def create_submission(labels, test_filenames, test_pred, submission_filename):
    test_path = os.path.join(globals.ROOT_PATH, "training", "images")
    with open(submission_filename, "w") as f:
        f.write("id,prediction\n")
        for fn, patch_array in zip(sorted(test_filenames), test_pred):
            img_number = int(re.search(r"\d+", fn).group(0))
            for i in range(patch_array.shape[0]):
                for j in range(patch_array.shape[1]):
                    f.write(
                        "{:03d}_{}_{},{}\n".format(
                            img_number,
                            j * globals.PATCH_SIZE,
                            i * globals.PATCH_SIZE,
                            int(patch_array[i, j]),
                        )
                    )


def np_to_tensor(x, device):
    # allocates tensors from np.arrays
    if device == "cpu":
        return torch.from_numpy(x).cpu()
    else:
        return torch.from_numpy(x).contiguous().pin_memory().to(device=device, non_blocking=True)


def show_val_samples(x, y, y_hat, segmentation=False):
    # training callback to show predictions on validation set
    imgs_to_draw = min(5, len(x))
    if x.shape[-2:] == y.shape[-2:]:  # segmentation
        fig, axs = plt.subplots(3, imgs_to_draw, figsize=(18.5, 12))
        for i in range(imgs_to_draw):
            axs[0, i].imshow(np.moveaxis(x[i], 0, -1))
            axs[1, i].imshow(np.concatenate([np.moveaxis(y_hat[i], 0, -1)] * 3, -1))
            axs[2, i].imshow(np.concatenate([np.moveaxis(y[i], 0, -1)] * 3, -1))
            axs[0, i].set_title(f"Sample {i}")
            axs[1, i].set_title(f"Predicted {i}")
            axs[2, i].set_title(f"True {i}")
            axs[0, i].set_axis_off()
            axs[1, i].set_axis_off()
            axs[2, i].set_axis_off()
    else:  # classification
        fig, axs = plt.subplots(1, imgs_to_draw, figsize=(18.5, 6))
        for i in range(imgs_to_draw):
            axs[i].imshow(np.moveaxis(x[i], 0, -1))
            axs[i].set_title(
                f"True: {np.round(y[i]).item()}; Predicted: {np.round(y_hat[i]).item()}"
            )
            axs[i].set_axis_off()
    plt.show()


def train(train_dataloader, eval_dataloader, model, loss_fn, metric_fns, optimizer, nr_epochs, patience=10, model_name="model", checkpoint_dir="checkpoints"):
    # training loop
    logdir = "./tensorboard/net"
    writer = SummaryWriter(logdir)  # tensorboard writer (can also log images)
    initial_lr = optimizer.param_groups[0]["lr"]
    history = {}  # collects metrics at the end of each epoch
    last_epoch = 0
    early_stopping = EarlyStopping(patience=patience, verbose=True, model_name=model_name, checkpoint_dir=checkpoint_dir)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=patience//2)
    for epoch in range(nr_epochs):  # loop over the dataset multiple times
        last_epoch = epoch
        # initialize metric list
        metrics = {"loss": [], "val_loss": []}
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics["val_" + k] = []

        pbar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{nr_epochs}")
        # training
        model.train()
        for x, y in pbar:
            optimizer.zero_grad()  # zero out gradients
            y_hat = model(x)  # forward pass
            loss = loss_fn(y_hat, y)
            loss.backward()  # backward pass
            optimizer.step()  # optimize weights

            # log partial metrics
            metrics["loss"].append(loss.item())
            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y).item())
            pbar.set_postfix({k: sum(v) / len(v) for k, v in metrics.items() if len(v) > 0})

        # validation
        
        model.eval()
        with torch.no_grad():  # do not keep track of gradients
            for x, y in eval_dataloader:
                y_hat = model(x)  # forward pass
                loss = loss_fn(y_hat, y)

                # log partial metrics
                metrics["val_loss"].append(loss.item())
                for k, fn in metric_fns.items():
                    metrics["val_" + k].append(fn(y_hat, y).item())

        # summarize metrics, log to tensorboard and display
        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
        for k, v in history[epoch].items():
            writer.add_scalar(k, v, epoch)
        logger.info(
            " ".join(
                ["\t- " + str(k) + " = " + str(v) + "\n " for (k, v) in history[epoch].items()]
            )
        )
        scheduler.step(history[epoch]["val_patch_f1"])
        if epoch % 5 == 0:
            print(f"Last lr: {scheduler.optimizer.param_groups[0]['lr']}")
        early_stopping(history[epoch]["val_patch_f1"], model)
        if early_stopping.early_stop:
            break

    
    # reset lr
    optimizer.param_groups[0]["lr"] = initial_lr
    
    model.load_state_dict(torch.load(early_stopping.best_checkpoint))
    logger.info(f"Restoring best checkpoint: {early_stopping.best_checkpoint} with F1 score: {early_stopping.best_score} after {last_epoch} epochs. Clearing checkpoint...")
    logger.info(
        "Finished Training with : "
        + " ".join(
            ["\t- " + str(k) + " = " + str(v) + "\n " for (k, v) in history[last_epoch].items()]
        )
    )
    early_stopping.clear_checkpoint()
    print("Finished Training")
    # plot loss curves
    plt.plot([v["loss"] for k, v in history.items()], label="Training Loss")
    plt.plot([v["val_loss"] for k, v in history.items()], label="Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title(f"Training and Validation Loss for {model_name}")
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, f"{model_name}-loss-{str([v['loss'] for k, v in history.items()][-1])}.png"))
    (globals.SHOW_PLOTS and plt.show()) or plt.clf()

    # save plot along logs for future reference
    # plot f1 curves
    plt.plot([v["f1"] for k, v in history.items()], label="Training F1")
    plt.plot([v["val_f1"] for k, v in history.items()], label="Validation F1")
    plt.ylabel("F1")
    plt.xlabel("Epochs")
    plt.title(f"F1 Score for {model_name}")
    plt.legend()
    plt.savefig(os.path.join(LOG_DIR, f"{model_name}-f1-{str([v['f1'] for k, v in history.items()][-1])}.png"))
    globals.SHOW_PLOTS and plt.show()
    # save plot along logs for future reference

def evaluate(eval_dataloader, model, loss_fn, metric_fns, nr_epochs, ):
    # training loop
    history = {}  # collects metrics at the end of each epoch
    for epoch in range(nr_epochs):  # loop over the dataset multiple times
        # initialize metric list
        metrics = {"val_loss": []}
        for k, _ in metric_fns.items():
            # metrics[k] = []
            metrics["val_" + k] = []
        # validation
        model.eval()
        with torch.no_grad():  # do not keep track of gradients
            for x, y in eval_dataloader:
                y_hat = model(x)  # forward pass
                loss = loss_fn(y_hat, y)

                # log partial metrics
                metrics["val_loss"].append(loss.item())
                for k, fn in metric_fns.items():
                    metrics["val_" + k].append(fn(y_hat, y).item())

        # summarize metrics, log to tensorboard and display
        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
    for k, v in history[epoch].items():
        print(
            " ".join(
                ["\t- " + str(k) + " = " + str(v) + "\n " for (k, v) in history[epoch].items()]
            )
        )


    print("Finished Training")
    # plot loss curves
    plt.plot([v["val_loss"] for k, v in history.items()], label="Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    (globals.SHOW_PLOTS and plt.show()) or plt.clf()

    # save plot along logs for future reference
    # plot f1 curves
    plt.plot([v["val_f1"] for k, v in history.items()], label="Validation F1")
    plt.ylabel("F1")
    plt.xlabel("Epochs")
    plt.legend()
    globals.SHOW_PLOTS and plt.show()
    # save plot along logs for future reference


def patch_accuracy_fn(y_hat, y):
    # computes accuracy weighted by patches (metric used on Kaggle for evaluation)
    h_patches = y.shape[-2] // globals.PATCH_SIZE
    w_patches = y.shape[-1] // globals.PATCH_SIZE
    patches_hat = (
        y_hat.reshape(-1, 1, h_patches, globals.PATCH_SIZE, w_patches, globals.PATCH_SIZE).mean(
            (-1, -3)
        )
        > globals.CUTOFF
    )
    patches = (
        y.reshape(-1, 1, h_patches, globals.PATCH_SIZE, w_patches, globals.PATCH_SIZE).mean(
            (-1, -3)
        )
        > globals.CUTOFF
    )
    return (patches == patches_hat).float().mean()


def accuracy_fn(y_hat, y):
    # computes classification accuracy
    return (y_hat.round() == y.round()).float().mean()


def f1_score_fn(y_hat, y):
    # computes the F1 score
    tp = (y_hat.round() * y).sum()
    fp = (y_hat.round() * (1 - y)).sum()
    fn = ((1 - y_hat.round()) * y).sum()
    return tp / (tp + 0.5 * (fp + fn))


def patch_f1_score_fn(y_hat, y):
    h_patches = y.shape[-2] // globals.PATCH_SIZE
    w_patches = y.shape[-1] // globals.PATCH_SIZE
    patches_hat = (
        y_hat.reshape(-1, 1, h_patches, globals.PATCH_SIZE, w_patches, globals.PATCH_SIZE).mean(
            (-1, -3)
        )
        > globals.CUTOFF
    ).int()
    patches = (
        y.reshape(-1, 1, h_patches, globals.PATCH_SIZE, w_patches, globals.PATCH_SIZE).mean(
            (-1, -3)
        )
        > globals.CUTOFF
    ).int()
    tp = (patches_hat * patches).sum()
    fp = (patches_hat * (1 - patches)).sum()
    fn = ((1 - patches_hat) * patches).sum()
    return tp / (tp + 0.5 * (fp + fn))


def show_patch_example(train_images, train_masks, val_images, val_masks):
    # extract all patches and visualize those from the first image
    train_patches, train_labels = utils.images_to_patches(train_images, train_masks)
    val_patches, val_labels = utils.images_to_patches(val_images, val_masks)
    utils.show_patched_image(train_patches[: 25 * 25], train_labels[: 25 * 25])
    logger.info(
        "{0:0.2f}".format(sum(train_labels) / len(train_labels) * 100)
        + "% of training patches are labeled as 1."
    )
    print(
        "{0:0.2f}".format(sum(val_labels) / len(val_labels) * 100)
        + "% of validation patches are labeled as 1."
    )


def create_test_submission(model, device=DEVICE, resize_to=globals.RESIZE_TO, submission_filename="submission.csv"):
    ########### PREDICT ON TEST SET ############
    test_path = os.path.join(globals.ROOT_PATH, "test", "images")
    test_filenames = glob(test_path + "/*.png")
    test_images = utils.load_all_from_path(test_path)
    batch_size = test_images.shape[0]
    size = test_images.shape[1:3]
    # resize the test images
    test_images = np.stack([cv2.resize(img, dsize=resize_to) for img in test_images], 0)
    test_images = test_images[:, :, :, :3]
    test_images = utils.np_to_tensor(np.moveaxis(test_images, -1, 1), device)
    # predict
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
        test_pred, test_filenames, test_pred, submission_filename=submission_filename
    )
    return model

def create_test_submission_sigmoid(model, device=DEVICE, resize_to=globals.RESIZE_TO, submission_filename="submission.csv"):
    ########### PREDICT ON TEST SET ############
    test_path = os.path.join(globals.ROOT_PATH, "test", "images")
    test_filenames = glob(test_path + "/*.png")
    test_images = utils.load_all_from_path(test_path)
    batch_size = test_images.shape[0]
    size = test_images.shape[1:3]
    # resize the test images
    test_images = np.stack([cv2.resize(img, dsize=resize_to) for img in test_images], 0)
    test_images = test_images[:, :, :, :3]
    test_images = utils.np_to_tensor(np.moveaxis(test_images, -1, 1), device)
    # predict
    test_pred = [torch.sigmoid(model(t)).detach().cpu().numpy() for t in test_images.unsqueeze(1)]
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
        test_pred, test_filenames, test_pred, submission_filename=submission_filename
    )
    return model

def get_ordered_checkpoints(prefix, directory="checkpoints"):
    checkpoints = [f for f in os.listdir(directory) if f.startswith(prefix)]
    if not checkpoints:
        return None
    checkpoints = sorted(checkpoints, reverse=True)
    checkpoints = [os.path.join(directory, f) for f in checkpoints]
    return checkpoints

def merge_dataloaders(*dataloaders):
    for itr in dataloaders:
        for v in itr:
            yield v

class MergeDataloaders:
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders
        self.itrs = [iter(itr) for itr in dataloaders]

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.itrs[0])
        except StopIteration:
            self.itrs.pop(0)
            if not self.itrs:
                self.itrs = [iter(itr) for itr in self.dataloaders]
                raise StopIteration
            return next(self)
    
    def __len__(self):
        return sum([len(itr) for itr in self.dataloaders])