from toolkit.logger import Logger

logger = Logger(__name__).get_logger()

import pandas as pd

import torch.cuda as cuda

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image

from torch import nn
from torch import squeeze
from torch.optim import Adam

from torchmetrics.classification import BinaryAccuracy

import gc


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
        image = read_image(self.images.loc[idx])
        label = self.labels.loc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class CustomImageDataLoader(dict):
    logger.debug(f"INIT: {__qualname__}")

    def __init__(
        self, dataset, image_col, label_col, transform=None, target_transform=None
    ):
        for k, v in dataset.items():
            self.update(
                {
                    f"{k}_data": CustomImageDataSet(
                        v[image_col],
                        v[label_col],
                        transform=transform,
                        target_transform=target_transform,
                    )
                }
            )
        logger.info("CustomImageDataSet created")

    def create_dataloaders(self, batch_size):
        tmp_dict = {
            k.split("_")[0]: DataLoader(v, batch_size=batch_size, shuffle=True)
            for k, v in self.items()
        }
        self.update(tmp_dict)
        logger.info("Dataloaders created")


class ModelResults:
    logger.debug(f"INIT: {__qualname__}")

    def __init__(self):
        self.data = pd.DataFrame(
            columns=["loss", "accuracy", "validation_loss", "validation_acc"]
        )

    def plot():
        pass


class NeuralNetwork(nn.Module):
    logger.debug(f"INIT: {__qualname__}")

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1),
        )
        self.loss = nn.BCELoss()
        self.optimizer = Adam(self.layers.parameters(), lr=1e-4)
        self.dev = None
        self.results = ModelResults()
        logger.info(self)

    def forward(self, x):
        x = self.layers(x)
        return nn.Sigmoid()(x)

    def init_device(self, device="auto"):
        logger.info("Setting up CUDA device")
        if device == "auto":
            if cuda.is_available():
                self.dev = "cuda"
                logger.info(f"CUDA set up to device: {cuda.get_device_name(self.dev)}")
            else:
                self.dev = "cpu"
                logger.warning(
                    f"Failed to set up CUDA, using: {cuda.get_device_name(self.dev)}"
                )
        else:
            self.dev = device
            logger.warning(f"Device set manually to {self.dev}")
        self.to(self.dev)
        logger.info(f"Model loaded to device: {cuda.get_device_name(self.dev)}")

    def train(self, epochs, train_loader, validation_loader=None):
        logger.info("Model training started")
        for epoch in range(epochs):
            res_epoch = pd.DataFrame(
                data={
                    "loss": 0.0,
                    "accuracy": 0.0,
                    "validation_loss": 0.0,
                    "validation_acc": 0.0,
                },
                index=[epoch],
            )

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.dev), labels.to(self.dev)
                outputs = self(inputs)
                loss = self.loss(squeeze(outputs), labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                res_epoch["loss"].loc[epoch] += loss.item()
                res_epoch["accuracy"].loc[epoch] += (
                    BinaryAccuracy()(squeeze(outputs).to("cpu"), labels.to("cpu"))
                    .detach()
                    .cpu()
                    .numpy()
                )

                del outputs, loss
                gc.collect()
                cuda.empty_cache()

            res_epoch["loss"].loc[epoch] /= len(train_loader)
            res_epoch["accuracy"].loc[epoch] /= len(train_loader)

            if validation_loader:
                for inputs, labels in validation_loader:
                    inputs, labels = inputs.to(self.dev), labels.to(self.dev)
                    outputs = self(inputs)
                    loss = self.loss(squeeze(outputs), labels)

                    res_epoch["validation_loss"].loc[epoch] += loss.item()
                    res_epoch["validation_acc"].loc[epoch] += (
                        BinaryAccuracy()(squeeze(outputs).to("cpu"), labels.to("cpu"))
                        .detach()
                        .cpu()
                        .numpy()
                    )

            res_epoch["validation_loss"].loc[epoch] /= len(validation_loader)
            res_epoch["validation_acc"].loc[epoch] /= len(validation_loader)

            self.results.data = pd.concat([self.results.data, res_epoch])

            logger.info(
                f"Epoch: {epoch:4d} "
                f"Loss: {res_epoch['loss'][epoch]:7.4f} "
                f"Accuracy {res_epoch['accuracy'][epoch]:7.4f}  "
                f"Validation loss: {res_epoch['validation_loss'][epoch]:7.4f}   "
                f"Validation accuracy: {res_epoch['validation_acc'][epoch]:7.4f}"
            )
