import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):

    def __init__(self, ds, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tgt_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tgt_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tgt_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_token = self.src_tokenizer.encode(src_text).ids
        dec_input_token = self.tgt_tokenizer.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_token) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_token) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_token, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ], dim=0)

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_token, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ], dim=0)

        label = torch.cat([
            torch.tensor(dec_input_token, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ], dim=0)

        # asinity check
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze().unsqueeze().int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & BilingualDataset.casual_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text

        }
    
    @staticmethod
    def casual_mask(size):
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).dtype(torch.int64)
        return mask



        
