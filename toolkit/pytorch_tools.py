from toolkit.logger import Logger

logger = Logger("pytorch_tools").get_logger()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datetime import datetime

import multiprocessing

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import IsolationForest
from sklearn.inspection import DecisionBoundaryDisplay

import torch
import torch.cuda as cuda

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image
from torchvision.transforms.functional import to_tensor
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

from typing import Iterable, Tuple, Union, List

from tqdm.notebook import tqdm


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

    def __getitem__(self, idx) -> Tuple[Tensor, int]:
        image = to_tensor(Image.open(self.images.loc[idx]))
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
        num_workers: int = -1,
    ) -> None:
        self.num_workers = num_workers
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
        if self.num_workers == -1:
            self.num_workers = multiprocessing.cpu_count()
            logger.info(f"Setting dataloader subprocesses to {self.num_workers}")
        else:
            logger.warning(
                f"Setting dataloader subprocesses manually to {self.num_workers}"
            )
        tmp_dict: dict = {
            k.split("_")[0]: DataLoader(
                v, batch_size=batch_size, shuffle=True, num_workers=self.num_workers
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
            if self.name in {"VGG19", "EfficientNetV2L"}:
                self.encoder = self.implemented_models[self.name](
                    weights=weights
                ).features
            elif self.name == "ResNet50":
                loaded_model = self.implemented_models[self.name](weights=weights)
                self.encoder = nn.Sequential(
                    loaded_model.conv1,
                    loaded_model.bn1,
                    loaded_model.relu,
                    loaded_model.maxpool,
                    loaded_model.layer1,
                    loaded_model.layer2,
                    loaded_model.layer3,
                    loaded_model.layer4,
                )
            else:
                raise KeyError(f"Model not implemented: {self.name}")
        except KeyError as e:
            logger.error(f"Model not implemented: {self.name}")
            raise NotImplementedError(f"Model not implemented: {self.name}") from e

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
            raise NotImplementedError(f"Model not implemented: {self.name}") from e

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)


class MatchFilters(nn.Module):
    logger.debug(f"INIT: {__qualname__}")
    implemented_models = {
        "VGG19": [512],
        "ResNet50": [2048, 1024, 512],
        "EfficientNetV2L": [1280, 896, 512],
    }

    def __init__(self, encoder: str, filters: List[int] = None) -> None:
        super(MatchFilters, self).__init__()
        try:
            self.filters = filters or self.implemented_models[encoder]
            logger.info(f"Using matching filters: {self.filters}")
        except KeyError as e:
            logger.error(f"Model not implemented: {self.name}")
            raise NotImplementedError(f"Model not implemented: {self.name}") from e

        self.layers = []
        if range(len(self.filters) - 1):
            self.layers.extend(
                nn.Conv2d(
                    in_channels=self.filters[i],
                    out_channels=self.filters[i + 1],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for i in range(len(self.filters) - 1)
            )
        else:
            self.layers.extend([nn.Identity()])
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class AutoEncoder(nn.Module):
    logger.debug(f"INIT: {__qualname__}")

    def __init__(
        self,
        name: str,
        encoder_name: str,
        encoder_weights: Union[bool, str],
        decoder_name: str = "VGG19",
    ) -> None:
        super(AutoEncoder, self).__init__()
        self.name = name

        self.encoder = Encoder(name=encoder_name, weights=encoder_weights)
        self.decoder = Decoder(name=decoder_name)
        self.match_filters = MatchFilters(encoder=encoder_name)

        self.loss_fn = nn.MSELoss()
        self.loss = 0
        self.optimizer = Adam(self.parameters(), lr=5e-6)
        self.epochs = 0
        self.dev: str = None
        self.path = Path(f"./results/{self.name}/")
        if self.path.exists():
            logger.debug(f"Folder exists: /{self.path}")
        else:
            self.path.mkdir(parents=True)
            logger.info(f"Folder created: /{self.path}")
        self.update_filename()
        self.results = AutoEncoderResults(name=self.name, filename=self.filename)

        logger.info(f"AutoEncoder constructed: {self.name}")
        logger.info(self)

    def forward(self, x: Tensor) -> Tensor:
        out = self.encoder(x)
        out = self.match_filters(out)
        out = self.decoder(out)
        return out

    def predict(
        self, images: List[str], transform: Compose, plot: bool = False
    ) -> List[Tensor]:
        out = []
        for i, image in enumerate(images):
            img_in = Image.open(image)
            img = transform(to_tensor(img_in))
            img = unsqueeze(img, 0)
            img = img.to(self.dev)
            self.eval()
            with torch.no_grad():
                out.append(self(img))
            if plot:
                fig, axs = plt.subplots(1, 2, figsize=(10, 3), dpi=400)
                axs[0].imshow(np.asarray(img_in))
                axs[1].imshow(out[-1].squeeze().permute(1, 2, 0).cpu().detach().numpy())
                for ax in axs:
                    ax.set_xticks([])
                    ax.set_yticks([])
                axs[0].set_ylabel(image.split("/")[-1])
                plt.show()
                fig.savefig(
                    self.filename.with_stem(
                        f"{self.filename.stem}_predict_{i}"
                    ).with_suffix(".png")
                )
        return out

    def train_net(
        self,
        epochs: int,
        train_loader: DataLoader,
        validation_loader: DataLoader = None,
        save_freq: int = 0,
    ) -> None:
        logger.info("Model training started")
        for epoch in range(self.epochs, self.epochs + epochs):
            res_epoch: pd.DataFrame = pd.DataFrame(
                data={
                    "loss": 0.0,
                    "validation_loss": 0.0,
                },
                index=[epoch],
            )
            self.train()
            for inputs, _ in tqdm(train_loader):
                inputs = inputs.to(self.dev)
                outputs = self(inputs)
                self.loss = self.loss_fn(squeeze(outputs), squeeze(inputs))
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

                res_epoch["loss"].loc[epoch] += self.loss.item()

            res_epoch["loss"].loc[epoch] /= len(train_loader)

            if validation_loader:
                self.eval()
                for inputs, _ in validation_loader:
                    inputs = inputs.to(self.dev)
                    with torch.no_grad():
                        outputs = self(inputs)
                        loss = self.loss_fn(squeeze(outputs), squeeze(inputs))

                    res_epoch["validation_loss"].loc[epoch] += loss.item()

            res_epoch["validation_loss"].loc[epoch] /= len(validation_loader)

            self.results.data = pd.concat([self.results.data, res_epoch])
            self.epochs += 1

            logger.info(
                f"Epoch: {epoch:4d} "
                f"Loss: {res_epoch['loss'][epoch]:7.4f} "
                f"Validation loss: {res_epoch['validation_loss'][epoch]:7.4f}   "
            )

            if save_freq and self.epochs % save_freq == 0:
                self.save()
                self.results.save()

    def calc_feature_vectors(
        self, dataset: pd.DataFrame, transform: Compose
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        feature_vectors = [[], [], [], []]
        loss = []
        self.eval()
        for _, row in tqdm(dataset.iterrows(), total=len(dataset.index)):
            img = to_tensor(Image.open(row.img))
            transformed_img = transform(img)
            transformed_img = torch.unsqueeze(transformed_img, 0)
            transformed_img = transformed_img.to(self.dev)
            with torch.no_grad():
                encoded = self.encoder(transformed_img)
                filter_matched = self.match_filters(encoded)
                decoded = self.decoder(filter_matched)
                loss.append(
                    self.loss_fn(
                        torch.squeeze(decoded), torch.squeeze(transformed_img)
                    ).item()
                )
            for i, res in enumerate(
                [transformed_img, encoded, filter_matched, decoded]
            ):
                feature_vectors[i].append(res.flatten().cpu().detach().numpy())
        return feature_vectors, loss

    def plot_feature_vectors(
        self, feature_vectors: List[np.ndarray], loss: np.ndarray
    ) -> None:
        pca = PCA(n_components=50)
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        fig, ax = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True, dpi=400)
        ax = ax.flatten()
        titles = ["inputs", "encoded", "filter_matched", "decoded"]
        for v, vector in enumerate(feature_vectors):
            vector = np.array(vector)
            res_pca = pca.fit_transform(vector)
            res_tsne = pd.DataFrame(tsne.fit_transform(res_pca))
            sns.scatterplot(
                data=res_tsne,
                x=0,
                y=1,
                hue=loss,
                palette="Purples",
                ax=ax[v],
                legend=False,
            )
            ax[v].set_xlabel(None)
            ax[v].set_ylabel(None)
            ax[v].set_title(titles[v])
        norm = plt.Normalize(min(loss), max(loss))
        sm = plt.cm.ScalarMappable(cmap="Purples", norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, location="bottom", aspect=50, pad=0.07)
        plt.show()
        fig.savefig(
            self.filename.with_stem(
                f"{self.filename.stem}_feature_vectors"
            ).with_suffix(".png")
        )
        fig, ax = plt.subplots(1, 1, figsize=(15, 6), dpi=400)
        sns.lineplot(loss, ax=ax)
        plt.show()
        fig.savefig(
            self.filename.with_stem(
                f"{self.filename.stem}_feature_vectors_loss"
            ).with_suffix(".png")
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

    def save(self) -> None:
        save(
            {
                "epoch": self.epochs,
                "loss": self.loss,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            self.filename,
        )
        logger.info(f"Model saved to: /{self.filename}")

    def load(self, model_file: str = None) -> None:
        try:
            checkpoint = load(model_file, map_location=self.dev)
            self.epochs = checkpoint["epoch"]
            self.loss = checkpoint["loss"]
            self.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.filename = Path(model_file)
            self.results.filename = Path(model_file).with_suffix(".csv")
            logger.info(f"Model loaded from: /{model_file}")
        except Exception as e:
            logger.error(f"Model load unsuccessful: /{model_file}")
            raise RuntimeError(f"Model load unsuccessful: /{model_file}") from e


class AutoEncoderResults:
    logger.debug(f"INIT: {__qualname__}")

    def __init__(self, name: str, filename: Path) -> None:
        self.data: pd.DataFrame = pd.DataFrame(
            columns=["loss", "validation_loss"],
        )
        self.data.index.name = "epochs"
        self.name = name
        self.filename = filename.with_suffix(".csv")
        logger.info(f"Results filename: /{self.filename}")

    def save(self) -> None:
        self.data.to_csv(self.filename, index_label="epochs")
        logger.info(f"Results saved to: /{self.filename}")

    def load(self, filename: str) -> None:
        try:
            self.data = pd.read_csv(filename, index_col="epochs")
            logger.info(f"Results loaded from /{filename}")
        except Exception as e:
            logger.error(f"Load results unsuccessful: /{filename}")
            raise RuntimeError("Load results unsuccessful: {filename}") from e

    def plot(self) -> None:
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
        fig.savefig(
            self.filename.with_stem(f"{self.filename.stem}_results").with_suffix(".png")
        )


class AnomalyDetector:
    logger.debug(f"INIT: {__qualname__}")

    def __init__(self, detector_type: str, n_jobs: int = -1) -> None:
        self.type = detector_type
        self.n_jobs = n_jobs
        self.initialize()

    def initialize(self) -> None:
        if self.type == "IsolationForest":
            self.pipeline = Pipeline(
                steps=[
                    ("scaler", Normalizer()),
                    ("detector", IsolationForest(n_jobs=self.n_jobs)),
                ]
            )

    def fit(self, X: pd.DataFrame) -> None:
        self.pipeline.fit(X=X)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        anomaly_scores = self.pipeline.decision_function(X=X)
        predictions = self.pipeline.predict(X=X)
        return (
            pd.DataFrame(
                data={
                    "outlier": predictions == -1,
                    "anomaly_score": anomaly_scores,
                }
            )
            .sort_values("anomaly_score")
            .reset_index(names=["img_idx"])
        )
