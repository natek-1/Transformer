import os
import sys
from pathlib import Path
from datasets import load_from_disk
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
            ds_raw = load_from_disk(str(self.config.data_path))
            
            # Build Tokenizers
            logging.info("Building tokenizers")
            src_tokenizer = self.get_or_build_tokenizer(ds_raw, self.config.lang_src)
            tgt_tokenizer = self.get_or_build_tokenizer(ds_raw, self.config.lang_tgt)
            
            # Split data
            logging.info("Splitting data into train and validation sets")
            train_ds_size = int(0.9 * len(ds_raw))
            val_ds_size = len(ds_raw) - train_ds_size
            train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

            # Save split datasets
            train_dir = os.path.join(self.config.root_dir, "train")
            val_dir = os.path.join(self.config.root_dir, "val")
            
            # We need to save the subset or the dataset. 
            # random_split returns a Subset, which doesn't have save_to_disk directly in older versions, 
            # but usually we can process it. However, to be safe and standard with HF datasets:
            # We can select indices.
            
            # Actually, let's use the indices from random_split to create new datasets if needed, 
            # or just save the subsets if supported (Dataset.save_to_disk supports it in newer versions, 
            # but Subsets might not).
            # Let's verify if we can save subsets. If not, we'll convert to Dataset.
            # Safe way:
            import numpy as np
            indices = np.arange(len(ds_raw))
            # But random_split is random. 
            # Better approach for reproducibility and saving:
            # split method from datasets library is better than torch.content.random_split for HF datasets
            
            # Re-doing split using HF datasets method for easier saving
            ds_split = ds_raw.train_test_split(test_size=0.1) # 0.9 train, 0.1 test (val)
            train_ds_hf = ds_split['train']
            val_ds_hf = ds_split['test']
            
            logging.info(f"Saving train dataset to {train_dir}")
            train_ds_hf.save_to_disk(train_dir)
            
            logging.info(f"Saving val dataset to {val_dir}")
            val_ds_hf.save_to_disk(val_dir)

            src_tokenizer_path = Path(self.config.root_dir) / Path(str(self.config.tokenizer_file).format(self.config.lang_src))
            tgt_tokenizer_path = Path(self.config.root_dir) / Path(str(self.config.tokenizer_file).format(self.config.lang_tgt))

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=Path(train_dir),
                transformed_val_file_path=Path(val_dir),
                src_tokenizer_file_path=src_tokenizer_path,
                tgt_tokenizer_file_path=tgt_tokenizer_path
            )
            
            logging.info("Data transformation completed successfully")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)
