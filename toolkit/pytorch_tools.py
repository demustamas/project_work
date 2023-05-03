from toolkit.logger import Logger

logger = Logger("pytorch_tools").get_logger()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from jupyter_dash import JupyterDash
from dash import html
from dash import dcc
from dash import Input
from dash import Output
import plotly.express as px

from pathlib import Path
from datetime import datetime

import multiprocessing

import torch
import torch.cuda as cuda

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.models import vgg19
from torchvision.models import resnet50
from torchvision.models import efficientnet_v2_l
from torchvision.transforms import Compose

from torch import nn
from torch import squeeze
from torch import unsqueeze
from torch import Tensor
from torch import save
from torch import load
from torch.optim import Adam

import gc
from typing import Iterable, Tuple, Union


class CustomImageDataSet(Dataset):
    logger.debug(f"INIT: {__qualname__}")

    def __init__(
        self,
        images: Iterable[str],
        labels: Iterable[np.float32],
        transform: Compose = None,
        target_transform: Compose = None,
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

    def create_dataloaders(self, batch_size: int, num_workers: int = -1) -> None:
        if num_workers == -1:
            num_workers = multiprocessing.cpu_count()
            logger.info(f"Setting dataloader subprocesses to {num_workers}")
        else:
            logger.warning(f"Setting dataloader subprocesses manually to {num_workers}")
        tmp_dict: dict = {
            k.split("_")[0]: DataLoader(
                v, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )
            for k, v in self.items()
        }
        self.update(tmp_dict)
        logger.info("Dataloaders created")


class Encoder(nn.Module):
    logger.debug(f"INIT: {__qualname__}")
    implemented_models = {
        "VGG19": vgg19,
        "ResNet50": resnet50,
        "EfficientNetV2L": efficientnet_v2_l,
    }

    def __init__(self, name: str, weights: Union[bool, str] = None) -> None:
        super(Encoder, self).__init__()
        self.name = name
        try:
            self.encoder = self.implemented_models[self.name](weights=weights).features
        except KeyError as e:
            logger.error(f"Model not implemented: {self.name}")
            raise KeyError(f"Model not implemented: {self.name}") from e

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    logger.debug(f"INIT: {__qualname__}")
    implemented_models = {
        "VGG19": nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
            ),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=3, kernel_size=(3, 3), stride=1, padding=1
            ),
        )
    }

    def __init__(self, name: str) -> None:
        super(Decoder, self).__init__()
        self.name = name
        try:
            self.decoder = self.implemented_models[self.name]
        except KeyError as e:
            logger.error(f"Model not implemented: {self.name}")
            raise KeyError(f"Model not implemented: {self.name}") from e

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)


class NeuralNetwork(nn.Module):
    logger.debug(f"INIT: {__qualname__}")

    def __init__(
        self,
        name: str,
        num_classes: str,
        encoder_name: str,
        encoder_weights: Union[bool, str],
        decoder_name: str = "VGG19",
    ) -> None:
        super(NeuralNetwork, self).__init__()
        self.name = name
        self.num_classes = num_classes

        self.encoder = Encoder(name=encoder_name, weights=encoder_weights)
        self.decoder = Decoder(name=decoder_name)

        self.loss = nn.MSELoss()
        self.optimizer = Adam(self.parameters(), lr=5e-6)
        self.dev: str = None
        self.path = Path(f"./results/{self.name}/")
        if self.path.exists():
            logger.debug(f"Folder exists: {self.path}")
        else:
            self.path.mkdir(parents=True)
            logger.info(f"Folder created: {self.path}")
        self.update_filename()
        self.results = ModelResults(name=self.name, filename=self.filename)

        logger.info(f"Neural Network constructed: {self.name}")
        logger.info(self)

    def forward(self, x: Tensor) -> Tensor:
        out = self.encoder(x)
        out = self.decoder(out)
        return out

    def predict(self, img: str, transform: Compose) -> Tensor:
        img = read_image(img)
        transformed_img = transform(img)
        transformed_img = unsqueeze(transformed_img, 0)
        self.eval()
        with torch.no_grad():
            out = self(transformed_img)
        return out

    def train_net(
        self,
        epochs: int,
        train_loader: DataLoader,
        validation_loader: DataLoader = None,
    ) -> None:
        logger.info("Model training started")
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
            self.train()
            for inputs, _ in train_loader:
                inputs = inputs.to(self.dev)
                outputs = self(inputs)
                loss = self.loss(squeeze(outputs), squeeze(inputs))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                res_epoch["loss"].loc[epoch] += loss.item()
                self.clean_up([loss, outputs])

            res_epoch["loss"].loc[epoch] /= len(train_loader)

            if validation_loader:
                self.eval()
                for inputs, _ in validation_loader:
                    inputs = inputs.to(self.dev)
                    with torch.no_grad():
                        outputs = self(inputs)
                        loss = self.loss(squeeze(outputs), squeeze(inputs))

                    res_epoch["validation_loss"].loc[epoch] += loss.item()

                self.clean_up([loss, outputs])

            res_epoch["validation_loss"].loc[epoch] /= len(validation_loader)

            self.results.data = pd.concat([self.results.data, res_epoch])

            logger.info(
                f"Epoch: {epoch:4d} "
                f"Loss: {res_epoch['loss'][epoch]:7.4f} "
                f"Validation loss: {res_epoch['validation_loss'][epoch]:7.4f}   "
            )

    def update_filename(self, filename: Path = None) -> None:
        self.filename = filename or (
            self.path / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        )
        logger.info(f"Model filename: {self.filename}")

    def init_device(self, device: str = "auto") -> None:
        logger.info("Setting up CUDA device")
        if device == "auto":
            if cuda.is_available():
                self.dev = "cuda"
                logger.info(f"CUDA set up to device: {cuda.get_device_name(self.dev)}")
            else:
                self.dev = "cpu"
                logger.warning(f"Failed to set up CUDA, using: {self.dev}")
        else:
            self.dev = device
            logger.warning(f"Device set manually to {self.dev}")
        self.to(self.dev)

    def clean_up(self, variables: Iterable = None) -> None:
        for variable in variables:
            del variable
        gc.collect()
        cuda.empty_cache()

    def save_model(self, epoch: int = -1, loss: float = -1) -> None:
        if epoch == -1 or loss == -1:
            logger.warning("Epoch or Loss not given, model not saved!")
        else:
            save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss,
                },
                str(self.filename),
            )
            logger.info(f"Model saved to {self.filename}")

    def load_model(self, model_file: str = None) -> None:
        if model_file:
            try:
                checkpoint = load(model_file)
                self.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                logger.info(f"Model loaded from: {model_file}")
            except Exception:
                logger.warning(f"Model load unsuccessful: {model_file}")
        else:
            logger.warning("No model file given, model not loaded!")


class ModelResults:
    logger.debug(f"INIT: {__qualname__}")

    def __init__(self, name: str, filename: Path) -> None:
        self.data: pd.DataFrame = pd.DataFrame(
            columns=["loss", "accuracy", "validation_loss", "validation_acc"],
        )
        self.name = name
        self.filename = filename.with_suffix(".csv")
        self.dashboard: JupyterDash = None
        logger.info(f"Results filename: {self.filename}")

    def save_data(self) -> None:
        self.data.to_csv(self.filename)
        logger.info(f"Results saved to: {self.filename}")

    def load_data(self, filename: str) -> None:
        self.data = pd.read_csv(filename)
        logger.info(f"Results loaded from {filename}")

    def plot(self) -> None:
        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots(1, 1, figsize=(15, 5), dpi=400)
        sns.lineplot(
            data=self.data,
            x=self.data.index,
            y="loss",
            label="Train",
            ax=ax,
        )
        sns.lineplot(
            data=self.data,
            x=self.data.index,
            y="validation_loss",
            label="Validation",
            ax=ax,
        )
        fig.savefig(f"./tex_images/{self.name}_results.png")
