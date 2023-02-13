from toolkit.logger import Logger

logger = Logger(__name__).get_logger()

import pandas as pd
from pathlib import Path
import re
import gc


class DataFrameCreator:
    logger.debug(f"INIT: {__qualname__}")

    def __init__(self):
        self.columns = ["path", "filename", "file", "type", "category"]
        self.train = pd.DataFrame(columns=self.columns)
        self.validation = pd.DataFrame(columns=self.columns)
        self.test = pd.DataFrame(columns=self.columns)

    def __del__(self):
        logger.debug(f"DESTRUCT: {__name__}")
        gc.collect()

    def from_folder(self, folder, data_type):
        df = pd.DataFrame(columns=self.columns)
        folder = Path(f"./{folder}")
        for category in folder.iterdir():
            if category.is_dir():
                for data_file in category.iterdir():
                    if data_file.is_file():
                        new_entry = pd.DataFrame.from_dict(
                            {
                                "path": [data_file.parent],
                                "filename": [data_file.name],
                                "file": [data_file],
                                "type": [data_type],
                                "category": [str(data_file.parts[-2])],
                            }
                        )
                        df = pd.concat(
                            [
                                df,
                                new_entry,
                            ]
                        )
                    else:
                        logger.warning(f"Directory found under category: {data_file}")
            else:
                logger.warning(f"File found in folder: {category}")

        df.reset_index(inplace=True, drop=True)

        if data_type == "training":
            self.train = df.copy()
        elif data_type == "train":
            self.train = df.copy()
        elif data_type == "validation":
            self.validation = df.copy()
        elif data_type == "test":
            self.test = df.copy()
        else:
            logger.error("No such data type given!")

        logger.info(f"DataFrame created from {folder} as {data_type} data.")

    def load_dataset(self, root):
        self.root = Path(root)
        for data_type in self.root.iterdir():
            if data_type.is_dir():
                type_name = re.sub(
                    "[^0-9a-zA-Z.]+", "_", str(data_type.parts[-1]).lower()
                )
                self.from_folder(data_type, type_name)
            else:
                logger.warning(f"File found in folder: {data_type}")
        logger.info(f"Dataset loaded from {self.root} folder.")

    def info(self):
        datasets = {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }

        for k, v in datasets.items():
            print(f"Name :          {k}")
            print(f"Type:           {*v.type.unique(),}")
            print(f"Columns:        {*v.columns,}")
            print(f"Shape:          {v.shape}")
            print(f"Categories:     {*v.category.unique(),}")
            print(f"Path:           {*v.path.unique(),}")
            print(
                f"File types:     {*v.filename.apply(lambda x: str(x).split('.')[-1]).unique(),}"
            )
            print("")
