

from random import random, uniform
from torchvision.transforms import functional as F
import torchvision.transforms as transforms


def tensor_transforms(x, y):
    if random() < 0.5:
        x = F.hflip(x) # image
        y = F.hflip(y) # mask
    if random() < 0.5:
        x = F.vflip(x)
        y = F.vflip(y)
    if random() < 0.5:
        # 90 degree rotation
        x = F.rotate(x, 90)
        y = F.rotate(y, 90)
    # random zoom
    i, j, h, w = transforms.RandomResizedCrop.get_params(
        x, scale=(0.75, 1.0), ratio=(0.9, 1.1)
    )
    x = F.resized_crop(x, i, j, h, w, size=(x.shape[1], x.shape[2]))
    y = F.resized_crop(y, i, j, h, w, size=(y.shape[1], y.shape[2]))
    # random brightness, contrast, saturation
    x = F.adjust_brightness(x, uniform(0.8, 1.2))
    x = F.adjust_contrast(x, uniform(0.8, 1.2))
    
    x = F.adjust_saturation(x, uniform(0.8, 1.2))
    if random() < 0.66: # always apply affine
        degrees = uniform(-10, 10)
        translate = (uniform(-0.1, 0.1) * x.shape[1], uniform(-0.1, 0.1) * x.shape[2])
        shear = uniform(-10, 10)
        scale = uniform(1., 1.1)
        x = F.affine(x, degrees, translate, scale, shear, fill=tuple(x[:, 0, 0].tolist()))
        y = F.affine(y, degrees, translate, scale, shear, fill=tuple(y[:, 0, 0].tolist()))


    return x, y

def tensor_transforms_zoom(x, y):
    if random() < 0.5:
        x = F.hflip(x) # image
        y = F.hflip(y) # mask
    if random() < 0.5:
        x = F.vflip(x)
        y = F.vflip(y)
    # random zoom
    i, j, h, w = transforms.RandomResizedCrop.get_params(
        x, scale=(0.5, 0.7), ratio=(1.0, 1.0)
    )
    x = F.resized_crop(x, i, j, h, w, size=(x.shape[1], x.shape[2]))
    y = F.resized_crop(y, i, j, h, w, size=(y.shape[1], y.shape[2]))
    # random rotation
    # angle = random() * 6 - 3
    # x = F.rotate(x, angle)
    # random brightness, contrast, saturation
    # x = F.adjust_brightness(x, random() * 0.05 + 1)
    # x = F.adjust_contrast(x, random() * 0.05 + 1)
    # x = F.adjust_saturation(x, random())
    return x, y