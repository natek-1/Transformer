import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from datasets import load_from_disk
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

from transformer.entity.config_entity import ModelTrainerConfig
from transformer.entity.artifact_entity import ModelTrainerArtifact
from transformer.dataset.dataset import BilingualDataset
from transformer.model.model import build_transformer
from transformer.logger import logging
from transformer.exception import CustomException

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def get_model(self, vocab_src_len, vocab_tgt_len):
        model = build_transformer(
            src_vocab_size=vocab_src_len, 
            tgt_vocab_size=vocab_tgt_len, 
            src_seq_len=self.config.seq_len,
            tgt_seq_len=self.config.seq_len, 
            d_model=self.config.d_model
        )
        return model

    def greedy_decode(self, model, src_input, src_mask, tgt_tokenizer, max_len, device):
        sos_idx = tgt_tokenizer.token_to_id("[SOS]")
        eos_idx = tgt_tokenizer.token_to_id("[EOS]")

        # encoder output 
        encoder_output = model.encode(src_input, src_mask)
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(src_input).to(device)

        while decoder_input.size(-1) != max_len:

            decoder_mask = BilingualDataset.casual_mask(decoder_input.size(1)).type_as(src_mask).to(device)
            out = model.decode(encoder_output, src_mask, decoder_input, decoder_mask)

            prob = model.project(out[:,-1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(src_input).fill_(next_word.item()).to(device)], dim=1
            )

            if next_word == eos_idx:
                break
        
        return decoder_input.squeeze(0)

    def run_validation(self, model, device, validation_dataset, tgt_tokenizer, max_len, print_msg, global_step, writer, num_examples=2):
        model.eval()
        count = 0

        source_texts = []
        expected = []
        predicted = []
        
        console_width = 80
        
        with torch.no_grad():
            for batch in validation_dataset:
                count += 1
                encoder_input = batch["encoder_input"].to(device)
                encoder_mask = batch["encoder_mask"].to(device)

                model_out = self.greedy_decode(model, encoder_input, encoder_mask, tgt_tokenizer, max_len, device)
                
                src_text = batch["src_text"][0]
                tgt_text = batch["tgt_text"][0]
                model_out_text = tgt_tokenizer.decode(model_out.detach().cpu().numpy())

                source_texts.append(src_text)
                expected.append(tgt_text)
                predicted.append(model_out_text)

                print_msg('-'*console_width)
                print_msg(f"{f'SOURCE: ':>12}{src_text}")
                print_msg(f"{f'TARGET: ':>12}{tgt_text}")
                print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

                if count == num_examples:
                    print_msg('-'*console_width)
                    break
        
        if writer:
            # Metrics logging (simplified error handling)
            pass

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting Model Training")
            
            # 1. Load Data
            train_data_path = os.path.join(self.config.dataset_path, "train")
            val_data_path = os.path.join(self.config.dataset_path, "val")
            
            logging.info(f"Loading training data from {train_data_path}")
            train_ds_loaded = load_from_disk(train_data_path)
            logging.info(f"Loading validation data from {val_data_path}")
            val_ds_loaded = load_from_disk(val_data_path)
            
            # 2. Tokenizers
            tokenizer_path_src = Path(self.config.data_path) / f"tokenizer_{self.config.lang_src}.json"
            tokenizer_path_tgt = Path(self.config.data_path) / f"tokenizer_{self.config.lang_tgt}.json"
            
            logging.info(f"Loading tokenizers from {tokenizer_path_src} and {tokenizer_path_tgt}")
            src_tokenizer = Tokenizer.from_file(str(tokenizer_path_src))
            tgt_tokenizer = Tokenizer.from_file(str(tokenizer_path_tgt))
            
            # 3. Create Datasets and Dataloaders
            train_ds = BilingualDataset(train_ds_loaded, src_tokenizer, tgt_tokenizer, self.config.lang_src, self.config.lang_tgt, self.config.seq_len)
            val_ds = BilingualDataset(val_ds_loaded, src_tokenizer, tgt_tokenizer, self.config.lang_src, self.config.lang_tgt, self.config.seq_len)
            
            train_dataloader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True,
                                        num_workers=12, prefetch_factor=64)
            val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
            
            # 4. Model
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            logging.info(f"Using device: {device}")
            device = torch.device(device)
            
            model = self.get_model(src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size()).to(device)
            
            # 5. Training Setup
            optimizer = torch.optim.Adam(model.parameters(), lr=float(self.config.lr), eps=1e-9)
            loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
            
            # Tensorboard
            writer = SummaryWriter(log_dir=os.path.join(self.config.root_dir, "logs"))
            
            global_step = 0
            
            for epoch in range(self.config.num_epochs):
                model.train()
                batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
                
                for batch in batch_iterator:
                    encoder_input = batch["encoder_input"].to(device)
                    decoder_input = batch["decoder_input"].to(device)
                    encoder_mask = batch["encoder_mask"].to(device)
                    decoder_mask = batch["decoder_mask"].to(device) # (batch_size, 1, seq_len, seq_len)
                    
                    encoder_output = model.encode(encoder_input, encoder_mask)
                    decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                    proj_output = model.project(decoder_output)
                    
                    label = batch["label"].to(device)
                    
                    loss = loss_fn(proj_output.view(-1, tgt_tokenizer.get_vocab_size()), label.view(-1))
                    
                    batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
                    writer.add_scalar("train_loss", loss.item(), global_step)
                    
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    
                    global_step += 1
                
                # Validation
                self.run_validation(model, device, val_dataloader, tgt_tokenizer, self.config.seq_len, 
                                    lambda msg: batch_iterator.write(msg), global_step, writer)
                
                # Save Model
                model_path = os.path.join(self.config.root_dir, f"{self.config.model_name}")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step
                }, model_path)
                
            logging.info("Model training completed")
            
            return ModelTrainerArtifact(trained_model_file_path=Path(model_path))

        except Exception as e:
            raise CustomException(e, sys)
