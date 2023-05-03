from toolkit.logger import Logger

logger = Logger("classes").get_logger()

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from pathlib import Path
import gc

from typing import List, Dict


class DataFrameCreator(dict):
    logger.debug(f"INIT: {__qualname__}")

    def __init__(self) -> None:
        self.columns: List[str] = [
            "img_dir",
            "img_file",
            "img",
            "label",
            "label_enc",
        ]
        self.update(
            {
                "dataset": pd.DataFrame(columns=self.columns),
                "train": pd.DataFrame(columns=self.columns),
                "validation": pd.DataFrame(columns=self.columns),
                "test": pd.DataFrame(columns=self.columns),
            }
        )

    def __del__(self) -> None:
        logger.debug(f"DESTRUCT: {__name__}")
        gc.collect()

    def load_dataset(self, sample_dir: str, labels: Dict) -> None:
        for label, encode in labels.items():
            src = Path(sample_dir) / label
            for e, entity in enumerate(src.iterdir()):
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
        self["dataset"].sort_values(by=["img"], inplace=True)
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

    def info(self) -> None:
        for k, v in self.items():
            logger.info(f"Name :          {k}")
            logger.info(f"Labels:         {v.label.value_counts().to_dict()}")
            logger.info(f"Columns:        {*v.columns,}")
            logger.info(f"Shape:          {v.shape}")
            logger.info(
                f"File types:     {*v.img_file.apply(lambda x: str(x).split('.')[-1]).unique(),}"
            )
