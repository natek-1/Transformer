import os
import sys
from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from transformer.entity.config_entity import DataValidationConfig
from transformer.entity.artifact_entity import DataValidationArtifact
from transformer.logger import logging
from transformer.exception import CustomException


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def get_all_sentences(self, dataset, lang):
        for item in dataset:
            yield item['translation'][lang]

    def get_or_build_tokenizer(self, dataset, lang):
        """Build a temporary tokenizer for validation purposes"""
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(self.get_all_sentences(dataset, lang), trainer=trainer)
        return tokenizer

    def validate_data(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")
            
            # Load data
            ds_raw = load_from_disk(str(self.config.data_path))
            logging.info(f"Loaded dataset with {len(ds_raw)} examples")
            
            # Build tokenizers for validation
            logging.info("Building tokenizers for validation")
            src_tokenizer = self.get_or_build_tokenizer(ds_raw, self.config.lang_src)
            tgt_tokenizer = self.get_or_build_tokenizer(ds_raw, self.config.lang_tgt)
            
            # Find max lengths
            max_len_src = 0
            max_len_tgt = 0
            
            logging.info("Calculating maximum sequence lengths")
            for item in ds_raw:
                src_ids = src_tokenizer.encode(item['translation'][self.config.lang_src]).ids
                tgt_ids = tgt_tokenizer.encode(item['translation'][self.config.lang_tgt]).ids
                
                max_len_src = max(max_len_src, len(src_ids))
                max_len_tgt = max(max_len_tgt, len(tgt_ids))
            
            logging.info(f"Max length of source sentence: {max_len_src}")
            logging.info(f"Max length of target sentence: {max_len_tgt}")
            
            # Validate against configured seq_len
            validation_status = True
            message = "Data validation successful"
            
            if max_len_src > self.config.seq_len - 2:  # -2 for SOS and EOS tokens
                validation_status = False
                message = f"Source sentences exceed seq_len. Max: {max_len_src}, Configured: {self.config.seq_len - 2}"
                logging.warning(message)
            
            if max_len_tgt > self.config.seq_len - 2:
                validation_status = False
                message = f"Target sentences exceed seq_len. Max: {max_len_tgt}, Configured: {self.config.seq_len - 2}"
                logging.warning(message)
            
            if validation_status:
                logging.info(f"✓ All sequences fit within configured seq_len ({self.config.seq_len})")
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                max_src_length=max_len_src,
                max_tgt_length=max_len_tgt,
                message=message
            )
            
            logging.info("Data validation completed")
            return data_validation_artifact
            
        except Exception as e:
            raise CustomException(e, sys)
