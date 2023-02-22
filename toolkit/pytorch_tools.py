from toolkit.logger import Logger

logger = Logger(__name__).get_logger()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from jupyter_dash import JupyterDash
from dash import Dash
from dash import html
from dash import dcc
import plotly.express as px

from pathlib import Path
from datetime import datetime

import multiprocessing

import torch.cuda as cuda

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image

from torch import nn
from torch import squeeze
from torch import Tensor
from torch.optim import Adam

from torchmetrics.classification import BinaryAccuracy

import gc
from typing import Iterable, Callable, Tuple


class CustomImageDataSet(Dataset):
    logger.debug(f"INIT: {__qualname__}")

    def __init__(
        self,
        images: Iterable[str],
        labels: Iterable[np.float32],
        transform: Iterable[Callable] = None,
        target_transform: Iterable[Callable] = None,
    ):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[Tensor, np.float32]:
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
        self,
        dataset: dict,
        image_col: str,
        label_col: str,
        transform: dict = None,
        target_transform: dict = None,
    ) -> None:
        for k, v in dataset.items():
            self.update(
                {
                    f"{k}_data": CustomImageDataSet(
                        v[image_col],
                        v[label_col],
                        transform=transform.get(k),
                        target_transform=target_transform.get(k),
                    )
                }
            )
        logger.info("CustomImageDataSet created")

    def create_dataloaders(self, batch_size: int) -> None:
        num_workers: int = multiprocessing.cpu_count()
        logger.info(f"Setting dataloader subprocesses to {num_workers}")
        tmp_dict: dict = {
            k.split("_")[0]: DataLoader(
                v, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )
            for k, v in self.items()
        }
        self.update(tmp_dict)
        logger.info("Dataloaders created")


class ModelResults:
    logger.debug(f"INIT: {__qualname__}")

    def __init__(self, name: str) -> None:
        self.data: pd.DataFrame = pd.DataFrame(
            columns=["loss", "accuracy", "validation_loss", "validation_acc"]
        )
        self.name = name
        self.path = Path(f"./results/{self.name}/")
        self.update_filename()
        if self.path.exists():
            logger.debug(f"Folder exists: {self.path}")
        else:
            self.path.mkdir(parents=True)
            logger.info(f"Folder created: {self.path}")

    def save_data(self) -> None:
        self.data.to_csv(self.filename)
        logger.info(f"Results saved to: {self.filename}")

    def update_filename(self, filename: str = None) -> None:
        self.filename = filename or (
            self.path / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        logger.info(f"Results filename: {self.filename}")

    def plot(self) -> None:
        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots(1, 2, figsize=(15, 5), dpi=400)
        sns.lineplot(
            data=self.data,
            x=self.data.index,
            y="loss",
            label="Train",
            ax=ax[0],
        )
        sns.lineplot(
            data=self.data,
            x=self.data.index,
            y="validation_loss",
            label="Validation",
            ax=ax[0],
        )
        sns.lineplot(
            data=self.data,
            x=self.data.index,
            y="accuracy",
            label="Train",
            ax=ax[1],
        )
        sns.lineplot(
            data=self.data,
            x=self.data.index,
            y="validation_acc",
            label="Validation",
            ax=ax[1],
        )
        fig.savefig(f"./tex_images/{self.name}_results.png")

    def create_dashboard(self) -> None:
        self.app = JupyterDash(__name__)
        template = "plotly_dark"
        fig_loss = px.line(
            data_frame=self.data,
            x=self.data.index,
            y=["loss", "validation_loss"],
            temaplte=template,
        )
        fig_acc = px.line(
            data_frame=self.data,
            x=self.data.index,
            y=["accuracy", "validation_acc"],
            temaplte=template,
        )
        self.app.layout = html.Div(
            children=[
                html.H1(children="Railway Track Fault Detection"),
                html.H2(children="Loss functions"),
                dcc.Graph(id="loss", figure=fig_loss),
                html.H2(children="Accuracy functions"),
                dcc.Graph(id="loss", figure=fig_acc),
            ]
        )


class NeuralNetwork(nn.Module):
    logger.debug(f"INIT: {__qualname__}")

    def __init__(self, name: str) -> None:
        super(NeuralNetwork, self).__init__()
        self.name = name
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
        self.optimizer = Adam(self.layers.parameters(), lr=5e-6)
        self.dev: str = None
        self.results = ModelResults(name=self.name)

        logger.info(f"Neural Network constructed: {self.name}")
        logger.info(self)

    def forward(self, x: Tensor) -> Tensor:
        out = self.layers(x)
        return nn.Sigmoid()(out)

    def init_device(self, device: str = "auto") -> None:
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

    def clean_up(self, variables: Iterable = None) -> None:
        for variable in variables:
            del variable
        gc.collect()
        cuda.empty_cache()

    def train(
        self,
        epochs: int,
        train_loader: DataLoader,
        validation_loader: DataLoader = None,
    ) -> None:
        logger.info("Model training started")
        binary_accuracy = BinaryAccuracy().to(self.dev)
        for epoch in range(epochs):
            res_epoch: pd.DataFrame = pd.DataFrame(
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
                    binary_accuracy(squeeze(outputs).to(self.dev), labels.to(self.dev))
                    .detach()
                    .cpu()
                    .numpy()
                )

                self.clean_up([loss, outputs])

            res_epoch["loss"].loc[epoch] /= len(train_loader)
            res_epoch["accuracy"].loc[epoch] /= len(train_loader)

            if validation_loader:
                for inputs, labels in validation_loader:
                    inputs, labels = inputs.to(self.dev), labels.to(self.dev)
                    outputs = self(inputs)
                    loss = self.loss(squeeze(outputs), labels)

                    res_epoch["validation_loss"].loc[epoch] += loss.item()
                    res_epoch["validation_acc"].loc[epoch] += (
                        binary_accuracy(
                            squeeze(outputs).to(self.dev), labels.to(self.dev)
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )

                self.clean_up([loss, outputs])

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
