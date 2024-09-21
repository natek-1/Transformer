import os 
from pathlib import Path

import torch.backends
from tqdm import tqdm


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter



# Hugging face to import dataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from dataset import BilingualDataset
from model import build_transformer
from .config import get_config, get_weight_file_path, latest_weight_file_pat 

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

def get_dataset(config):

    ds_raw = load_dataset(f"{config["datasource"]}", f"{config['lang_src']}-{config['lang-tgt']}", train='train')

    # tokenizer
    src_tokenizer = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tgt_tokenizer = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # train val split
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, src_tokenizer, tgt_tokenizer, config['lang_src'], config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(val_ds_raw, src_tokenizer, tgt_tokenizer, config["lang_src"], config['lang_tgt'], config["seq_len"])

    max_len_src = 0
    max_len_tgt = 0

    # find max length
    for item in ds_raw:
        src_ids = src_tokenizer.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tgt_tokenizer.encode(item['translation'][config['tgt_src']]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer

def get_model(config, vocab_src_length, target_src_length):
    model = build_transformer(src_vocab_size=vocab_src_length, tgt_vocab_size=target_src_length, src_seq_len=config["seq_len"],
                              tgt_seq_len=config["seq_len"])
    return model



def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Using {device} for training")

    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")

    device = torch.device(device)

    # adding the pathfolder
    Path(f"{config["datasource"]}_{config["model_folder"]}").mkdir(parents=True, exist_ok=True)


    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = get_dataset(config)
    model = get_model(config, vocab_src_length=src_tokenizer.get_vocab_size(), target_src_length=tgt_tokenizer.get_vocab_size()).to(device)

    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 eps=1e-9,
                                 lr=config["lr"])
    
    initial_epoch = 0
    global_step=0
    preload = config["preload"]
    model_filename = latest_weight_file_pat(config) if preload == "lastest" else get_weight_file_path(config, preload) if preload else None
    if model_filename:
        print(f"Loading model from {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = 1 + state["epoch"]
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print("No model selected starting training from scratch")

    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id["[PAD]"], label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch: {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device) #(batch_size, seq_len)
            decoder_input = batch["decoder_input"].to(device) #(batch_size, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) #(B, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device) # (B, 1, seq_len, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch_size, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch_size, seq_len, vocab_size)

            label = batch["label"].to(device)# (batch_size, seq_len)

            # loss calculations
            loss = loss_fn(proj_output.view(-1, tgt_tokenizer.get_vocab_size(), label.view(-1)))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            # log the loss

            writer.add_scaler("train_loss", loss.item(), global_step)
            writer.flush()

            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1











