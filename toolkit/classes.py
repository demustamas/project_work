from toolkit.logger import Logger

logger = Logger(__name__).get_logger()

import pandas as pd
from pathlib import Path
import re
import gc


class DataFrameCreator(dict):
    logger.debug(f"INIT: {__qualname__}")

    def __init__(self):
        self.columns = ["path", "filename", "file", "type", "category", "cat_idx"]
        self.update(
            {
                "train": pd.DataFrame(columns=self.columns),
                "validation": pd.DataFrame(columns=self.columns),
                "test": pd.DataFrame(columns=self.columns),
            }
        )

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
                                "file": [str(data_file)],
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
        df["category"] = df["category"].astype("category")
        df["cat_idx"] = df["category"].cat.codes.astype("float32")
        df.reset_index(inplace=True, drop=True)

        self.update({data_type: df.copy()})

        logger.info(f"DataFrame created from {folder} as {data_type} data")

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
        logger.info(f"Dataset loaded from {self.root} folder")

    def info(self):
        for k, v in self.items():
            logger.info(f"Name :          {k}")
            logger.info(f"Type:           {*v.type.unique(),}")
            logger.info(f"Columns:        {*v.columns,}")
            logger.info(f"Shape:          {v.shape}")
            logger.info(f"Categories:     {*v.category.unique(),}")
            logger.info(f"Path:           {*v.path.unique(),}")
            logger.info(
                f"File types:     {*v.filename.apply(lambda x: str(x).split('.')[-1]).unique(),}"
            )
