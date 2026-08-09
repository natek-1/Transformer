"""
Inference module for the English -> French Transformer translator.

Loads the trained checkpoint (weightsv5_final_seq_len_update_required/tmodel_14.pt),
the source/target tokenizers, and exposes a single `translate(text)` function that
performs greedy decoding.
"""
import os
import threading

import torch
from tokenizers import Tokenizer

from transformer.model.model import build_transformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEQ_LEN = 512  # must match the checkpoint's positional encoding buffer size
D_MODEL = 512

CHECKPOINT_PATH = os.path.join(
    BASE_DIR, "artifacts", "model_trainer" ,"model.pt"
)
TOKENIZER_SRC_PATH = os.path.join(
    BASE_DIR, "artifacts", "data_transformation", "tokenizer_en.json"
)
TOKENIZER_TGT_PATH = os.path.join(
    BASE_DIR, "artifacts", "data_transformation", "tokenizer_fr.json"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Lazy-loaded globals (loaded once, on first use, thread-safe)
# ---------------------------------------------------------------------------
_model = None
_tokenizer_src = None
_tokenizer_tgt = None
_load_lock = threading.Lock()


def _load():
    """Load tokenizers + model weights once and cache them in module globals."""
    global _model, _tokenizer_src, _tokenizer_tgt

    if _model is not None:
        return

    with _load_lock:
        if _model is not None:  # re-check after acquiring the lock
            return

        if not os.path.isfile(TOKENIZER_SRC_PATH):
            raise FileNotFoundError(f"Source tokenizer not found at {TOKENIZER_SRC_PATH}")
        if not os.path.isfile(TOKENIZER_TGT_PATH):
            raise FileNotFoundError(f"Target tokenizer not found at {TOKENIZER_TGT_PATH}")
        if not os.path.isfile(CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

        tokenizer_src = Tokenizer.from_file(TOKENIZER_SRC_PATH)
        tokenizer_tgt = Tokenizer.from_file(TOKENIZER_TGT_PATH)

        model = build_transformer(
            src_vocab_size=tokenizer_src.get_vocab_size(),
            tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
            src_seq_len=SEQ_LEN,
            tgt_seq_len=SEQ_LEN,
            d_model=D_MODEL,
        )

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        model.eval()

        _model = model
        _tokenizer_src = tokenizer_src
        _tokenizer_tgt = tokenizer_tgt


def _causal_mask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int64)
    return mask == 0


@torch.no_grad()
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while decoder_input.size(1) < max_len:
        decoder_mask = _causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
            dim=1,
        )

        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)


def translate(text: str) -> str:
    """Translate a single sentence/paragraph of English text to French."""
    text = (text or "").strip()
    if not text:
        return ""

    _load()

    pad_idx = _tokenizer_src.token_to_id("[PAD]")
    sos_idx = _tokenizer_src.token_to_id("[SOS]")
    eos_idx = _tokenizer_src.token_to_id("[EOS]")

    enc_input_tokens = _tokenizer_src.encode(text).ids

    max_content_len = SEQ_LEN - 2  # reserve room for [SOS] and [EOS]
    if len(enc_input_tokens) > max_content_len:
        enc_input_tokens = enc_input_tokens[:max_content_len]

    num_padding_tokens = SEQ_LEN - len(enc_input_tokens) - 2

    encoder_input = torch.cat(
        [
            torch.tensor([sos_idx], dtype=torch.int64),
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            torch.tensor([eos_idx], dtype=torch.int64),
            torch.tensor([pad_idx] * num_padding_tokens, dtype=torch.int64),
        ],
        dim=0,
    ).unsqueeze(0).to(DEVICE)  # (1, seq_len)

    encoder_mask = (encoder_input != pad_idx).unsqueeze(0).unsqueeze(0).int().to(DEVICE)  # (1, 1, 1, seq_len)

    model_out = greedy_decode(
        _model, encoder_input, encoder_mask, _tokenizer_src, _tokenizer_tgt, SEQ_LEN, DEVICE
    )

    translation = _tokenizer_tgt.decode(model_out.detach().cpu().numpy().tolist())
    return translation


def warm_up():
    """Force model + tokenizers to load eagerly (e.g. at server startup)."""
    _load()


if __name__ == "__main__":
    import sys

    sample = " ".join(sys.argv[1:]) or "Hello, how are you today?"
    print(f"EN: {sample}")
    print(f"FR: {translate(sample)}")
