from toolkit.logger import Logger

logger = Logger("classes").get_logger()

import numpy as np
import pandas as pd
from pathlib import Path
import gc

from typing import List


class DataFrameCreator(dict):
    logger.debug(f"INIT: {__qualname__}")

    def __init__(self) -> None:
        self.columns: List[str] = [
            "img_path",
            "img_filename",
            "img",
            "label_path",
            "label_filename",
            "label",
            "type",
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

    def load_dataset(self, dataset: str, labels: str) -> None:
        for e, entity in enumerate(Path(dataset).iterdir()):
            if entity.is_file():
                label_img = Path(labels) / entity.name

                if label_img.exists():
                    df = pd.DataFrame(
                        {
                            "img_path": entity.parent,
                            "img_filename": entity.name,
                            "img": entity,
                            "label_path": label_img.parent,
                            "label_filename": label_img.name,
                            "label": label_img,
                            "type": "dataset",
                        },
                        index=[e],
                    )
                    self["dataset"] = pd.concat([self["dataset"], df], axis=0)
                else:
                    logger.warning(f"No label found for {entity}")
            elif entity.is_dir():
                logger.warning(f"Directory found: {entity}")
            else:
                logger.critical(f"Unknown entity type: {entity}")
                raise TypeError(f"Unknown entity type: {entity}")
        self["dataset"].sort_values(by=["img"], inplace=True)
        self["dataset"].reset_index(drop=True, inplace=True)
        logger.info(f"Dataset loaded from {dataset} with labels from {labels}")

    def split_dataset(self, val_rate: float = 0.2, test_rate: float = 0.1) -> None:
        idx = self["dataset"].index.to_list()
        val_idx = np.random.choice(idx, size=int(val_rate * len(idx)), replace=False)
        idx = [i for i in idx if i not in val_idx]
        test_idx = np.random.choice(idx, size=int(test_rate * len(idx)), replace=False)
        idx = [i for i in idx if i not in test_idx]
        self["train"] = self["dataset"].iloc[idx].copy()
        self["validation"] = self["dataset"].iloc[val_idx].copy()
        self["test"] = self["dataset"].iloc[test_idx].copy()
        self["train"].reset_index(drop=True, inplace=True)
        self["validation"].reset_index(drop=True, inplace=True)
        self["test"].reset_index(drop=True, inplace=True)

    def info(self) -> None:
        for k, v in self.items():
            logger.info(f"Name :          {k}")
            logger.info(f"Type:           {*v.type.unique(),}")
            logger.info(f"Columns:        {*v.columns,}")
            logger.info(f"Shape:          {v.shape}")
            logger.info(
                f"File types:     {*v.img_filename.apply(lambda x: str(x).split('.')[-1]).unique(),}"
            )
