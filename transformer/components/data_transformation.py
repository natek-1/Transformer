import os
import sys
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader, random_split

from transformer.entity.config_entity import DataTransformationConfig
from transformer.entity.artifact_entity import DataTransformationArtifact
from transformer.logger import logging
from transformer.exception import CustomException

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_all_sentences(self, dataset, lang):
        for item in dataset:
            yield item['translation'][lang]

    def get_or_build_tokenizer(self, dataset, lang):
        tokenizer_path = Path(self.config.root_dir) / Path(str(self.config.tokenizer_file).format(lang))
        if not Path.exists(tokenizer_path):
            os.makedirs(self.config.root_dir, exist_ok=True) # make directory if it doesn't exist
            tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
            tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
            tokenizer.train_from_iterator(self.get_all_sentences(dataset, lang), trainer=trainer)
            tokenizer.save(str(tokenizer_path))
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        return tokenizer

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            
            # Load data from ingestion artifact
            ds_raw = load_dataset('json',
                                data_files=f'opus_books_{self.config.lang_src}_{self.config.lang_tgt}.json',
                                split = 'train')
            
            # Build Tokenizers (and save them)
            logging.info("Building tokenizers")
            _ = self.get_or_build_tokenizer(ds_raw, self.config.lang_src)
            _ = self.get_or_build_tokenizer(ds_raw, self.config.lang_tgt)
            
            # Split data
            logging.info("Splitting data into train and validation sets")
            ds_split = ds_raw.train_test_split(test_size=0.1) # 0.9 train, 0.1 test (val)
            train_ds_hf = ds_split['train']
            val_ds_hf = ds_split['test']

            os.makedirs(self.config.dataset_path, exist_ok=True)
            train_path = os.path.join(self.config.dataset_path, "train")
            val_path = os.path.join(self.config.dataset_path, "val")
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(val_path, exist_ok=True)
            logging.info(f"Saving train dataset to {train_path}")
            train_ds_hf.save_to_disk(train_path)
            
            logging.info(f"Saving val dataset to {val_path}")
            val_ds_hf.save_to_disk(val_path)

            src_tokenizer_path = Path(self.config.root_dir) / Path(str(self.config.tokenizer_file).format(self.config.lang_src))
            tgt_tokenizer_path = Path(self.config.root_dir) / Path(str(self.config.tokenizer_file).format(self.config.lang_tgt))

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=Path(train_path),
                transformed_val_file_path=Path(val_path),
                src_tokenizer_file_path=src_tokenizer_path,
                tgt_tokenizer_file_path=tgt_tokenizer_path
            )
            
            logging.info("Data transformation completed successfully")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)
