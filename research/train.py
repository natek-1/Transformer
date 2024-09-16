import os 
from pathlib import Path

from tqdm import tqdm


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR



# Hugging face to import dataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, dataset, lang):
    tokenizer_path = Path(config["tokenizer_file".format(lang)])
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='["UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens= ["[UNK], [PAD], [SOS], [EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer