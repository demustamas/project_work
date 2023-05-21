from toolkit.logger import Logger

logger = Logger("classes").get_logger()

import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split

from pathlib import Path
import gc

from tqdm.notebook import tqdm

from typing import List, Dict


class ImageDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return ImageDataFrame

    def show_images(self, idxs: List[int]) -> None:
        for idx in idxs:
            _, ax = plt.subplots(1, 1, figsize=(15, 10), tight_layout=True, dpi=400)
            ax.imshow(Image.open(self.iloc[idx].img))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(f"{idx}")
            plt.show()


class DataFrameCreator(dict):
    logger.debug(f"INIT: {__qualname__}")
    columns: List[str] = [
        "img_dir",
        "img_file",
        "img",
        "label",
        "label_enc",
    ]

    def __init__(self) -> None:
        self.update(
            {
                "dataset": ImageDataFrame(columns=self.columns),
                "train": ImageDataFrame(columns=self.columns),
                "validation": ImageDataFrame(columns=self.columns),
                "test": ImageDataFrame(columns=self.columns),
            }
        )

    def __del__(self) -> None:
        logger.debug(f"DESTRUCT: {__name__}")
        gc.collect()

    def load_dataset(self, sample_dir: str, labels: Dict) -> None:
        for label, encode in labels.items():
            src = Path(sample_dir) / label
            for e, entity in tqdm(
                enumerate(src.iterdir()), total=len(list(src.rglob("*")))
            ):
                if entity.is_file():
                    df = pd.DataFrame(
                        {
                            "img_dir": str(entity.parent),
                            "img_file": str(entity.name),
                            "img": str(entity),
                            "label": str(label),
                            "label_enc": int(encode),
                        },
                        index=[e],
                    )
                    self["dataset"] = pd.concat([self["dataset"], df], axis=0)
                elif entity.is_dir():
                    logger.warning(f"Directory found: {entity}")
                else:
                    logger.critical(f"Unknown entity type: {entity}")
                    raise TypeError(f"Unknown entity type: {entity}")
            logger.info(f"Images loaded from {sample_dir} with label {label}")
        self["dataset"].sort_values(by=["img_file"], inplace=True)
        self["dataset"].reset_index(drop=True, inplace=True)

    def split_dataset(self, val_rate: float = 0.2, test_rate: float = 0.1) -> None:
        self["train"], _X = train_test_split(
            self["dataset"],
            test_size=val_rate + test_rate,
            shuffle=True,
            stratify=self["dataset"].label,
        )
        self["validation"], self["test"] = train_test_split(
            _X,
            test_size=test_rate / (val_rate + test_rate),
            shuffle=True,
            stratify=_X.label,
        )
        self["train"].reset_index(drop=True, inplace=True)
        self["validation"].reset_index(drop=True, inplace=True)
        self["test"].reset_index(drop=True, inplace=True)

    def info(self) -> None:
        for k, v in self.items():
            logger.info(f"Name :          {k}")
            logger.info(f"Labels:         {v.label.value_counts().to_dict()}")
            logger.info(f"Columns:        {*v.columns,}")
            logger.info(f"Shape:          {v.shape}")
            logger.info(
                f"File types:     {*v.img_file.apply(lambda x: str(x).split('.')[-1]).unique(),}"
            )
