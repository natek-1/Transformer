import os
import sys

from datasets import load_dataset

from transformer.entity.config_entity import DataIngestionConfig
from transformer.logger import logging
from transformer.exception import CustomException


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
    def download_data(self):
        if not os.path.exists(self.config.save_dir):
            ds_raw = load_dataset(f"{self.config.datasource}", f"{self.config.lang_src}-{self.config.lang_tgt}", split='train')
            ds_raw.to_json(self.config.save_dir)
            logging.info(f"{self.config.datasource} download! It was downloaded to the path {self.config.save_dir}")
        else:
            logging.info(f"File already exists of size: {self.config.save_dir}")  