from toolkit.logger import Logger

logger = Logger(__name__).get_logger()

import pandas as pd

import torch.cuda as cuda

from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomImageDataSet(Dataset):
    logger.debug(f"INIT: {__qualname__}")

    def __init__(self, images, labels, transform=None, target_transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = read_image(self.images.iloc[idx])
        label = self.labels.iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def setup_cuda():
    logger.debug(f"INIT: {__name__}")

    if cuda.is_available():
        dev = "cuda"
        logger.info(f"CUDA set up properly: {cuda.get_device_name(dev)}.")
    else:
        dev = "cpu"
        logger.warning(f"Failed to set up CUDA: {cuda.get_device_name(dev)}.")

    return dev
