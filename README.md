# Transformer

A from-scratch PyTorch implementation of the Transformer architecture from
["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), trained as an
English → French translation model, with a modular training pipeline and a
web UI for interactive translation.

## Contents

- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the training pipeline](#running-the-training-pipeline)
- [Known issue: data path mismatch](#known-issue-data-path-mismatch)
- [Running the web UI](#running-the-web-ui)
- [Translating from the command line](#translating-from-the-command-line)
- [Logs and artifacts](#logs-and-artifacts)
- [Citation](#citation)

## Architecture

`transformer/model/model.py` implements the encoder-decoder Transformer
from the paper, built from scratch (no `nn.Transformer`):

- **Input embeddings** scaled by `sqrt(d_model)`.
- **Sinusoidal positional encoding**, precomputed for a fixed `seq_len` and
  added to the embeddings (not learned).
- **Multi-head self-/cross-attention**, with Q/K/V/output projections as
  `bias=False` linear layers.
- **Layer normalization** with learnable `alpha`/`bias` (this repo's own
  implementation, not `nn.LayerNorm`), applied via pre-norm residual
  connections (`ResidualConnection`).
- **Position-wise feed-forward blocks** (`Linear → ReLU → Dropout → Linear`).
- 6 encoder blocks + 6 decoder blocks by default (`N=6`), 8 attention heads
  (`h=8`), `d_ff=2048`.
- A final **projection layer** maps decoder output to target vocabulary
  logits.

`build_transformer(...)` assembles all of this given source/target
vocabulary sizes and sequence length, and Xavier-initializes all
multi-dimensional parameters.

Note the **positional encoding buffers are sized to a fixed `seq_len` at
build time**, so a checkpoint trained with one `seq_len` cannot be loaded
into a model built with a different `seq_len` — the buffer shapes won't
match. This is why the seq_len used at inference must match the seq_len
used at training (see [Running the web UI](#running-the-web-ui)).

## Repository layout

- **`transformer/model/model.py`** — the Transformer architecture (see
  above).
- **`transformer/`** — the modular training pipeline, config-driven via
  `config/config.yaml` and `config/params.yaml`:
  - `transformer/components/data_ingestion.py`
  - `transformer/components/data_validation.py`
  - `transformer/components/data_transformation.py`
  - `transformer/components/model_trainer.py`
  - `transformer/pipline/training_pipeline.py` — orchestrates all four
    stages in order.
  - `transformer/dataset/dataset.py` — `BilingualDataset`: tokenizes,
    pads/truncates to `seq_len`, builds encoder/decoder masks (including the
    decoder's causal mask).
  - `transformer/entity/`, `transformer/configuration/` — dataclasses and
    the `ConfigurationManager` that reads the YAML configs into typed
    config objects.
  - `transformer/logger/`, `transformer/exception/` — logging (writes
    timestamped logs to `logs/`) and a custom exception wrapper used
    throughout the pipeline components.
- **`translate.py`** — loads a trained checkpoint + tokenizers once and
  exposes `translate(text)`, implementing **greedy decoding** for
  inference. Used by both the CLI and the web app.
- **`app.py`** + **`templates/index.html`** + **`static/style.css`** — a
  Flask web UI (English → French), styled like DeepL's translator, that
  calls `translate.py` under the hood.
- **`config/config.yaml`**, **`config/params.yaml`** — pipeline and model
  configuration (see [Configuration](#configuration)).
- **`weightsv5_final_seq_len_update_required/`** — checkpoints from the
  currently shipped training run (`tmodel_00.pt` … `tmodel_14.pt`), trained
  at `seq_len=350`.
- **`tokenizer_en.json`**, **`tokenizer_fr.json`** — word-level tokenizers
  matching the shipped checkpoint, used by `translate.py`.
- **`research/`**, **`train_*.py`**, **`load_model.py`**,
  **`opus_books_weights/`** — earlier/experimental standalone scripts and
  weights kept for reference. They predate the pipeline architecture and
  aren't required for training or serving the model; **the supported way
  to train is the pipeline** described below.

## How it works

The model is trained on the [`opus_books`](https://huggingface.co/datasets/opus_books)
English-French dataset. Text is tokenized with a word-level tokenizer
(vocab built per language, special tokens `[UNK]`/`[PAD]`/`[SOS]`/`[EOS]`),
and the Transformer is trained to predict the French translation
token-by-token given the English source, using teacher forcing and a
causal mask on the decoder side.

At inference time, translation uses **greedy decoding**: starting from
`[SOS]`, the model generates one token at a time, always picking the
highest-probability next token, feeding it back in, until it produces an
`[EOS]` token or hits `seq_len`.

### Pipeline stages

1. **Data ingestion** (`transformer/components/data_ingestion.py`) —
   downloads the `opus_books` en-fr dataset via HuggingFace `datasets` and
   writes it as newline-delimited JSON to `artifacts/opus_books`
   (`config.data_ingestion.save_dir`). Skipped if that path already
   exists.
2. **Data validation** (`transformer/components/data_validation.py`) —
   builds temporary word-level tokenizers over the full dataset and checks
   that every sentence's token count fits within `seq_len - 2` (room for
   `[SOS]`/`[EOS]`). Logs a warning (doesn't stop the pipeline) if some
   sequences would be truncated.
3. **Data transformation** (`transformer/components/data_transformation.py`) —
   builds and saves the real word-level tokenizers to
   `artifacts/data_transformation/tokenizer_en.json` /
   `tokenizer_fr.json`, splits the dataset 90/10 into train/val
   (`datasets.train_test_split`), and saves both splits (as HF `Dataset`
   objects, via `save_to_disk`) to `artifacts/data/train` and
   `artifacts/data/val`.
4. **Model training** (`transformer/components/model_trainer.py`) —
   builds the Transformer via `build_transformer` using
   `params.model_config` (`seq_len`, `d_model`), trains with Adam and
   `CrossEntropyLoss` (label smoothing 0.1, ignoring `[PAD]`), runs
   greedy-decode validation on 2 sample sentences each epoch, and
   checkpoints (`model_state_dict`, `optimizer_state_dict`, `epoch`,
   `global_step`) to `artifacts/model_trainer/model.pt`, **overwriting the
   same file every epoch** (so `model.pt` always holds the latest epoch —
   snapshot it elsewhere if you want to keep every epoch, as
   `weightsv5_final_seq_len_update_required/` does with its
   `tmodel_NN.pt` naming).

All stages are orchestrated in order by
`transformer/pipline/training_pipeline.py`. Training automatically uses
CUDA if available, then MPS (Apple Silicon), then falls back to CPU.

## Setup

Requires Python 3.9+ (see note below for Apple Silicon / M1).

```bash
conda create python=3.9 -n transformer -y
conda activate transformer
pip install -r requirements.txt
```

> If you're on an M1/Apple Silicon Mac and running `train.py`, use the
> conda setup above first — some packages need it to build correctly.

`requirements.txt` includes `-e .`, which installs this repo itself (via
`setup.py`) as an editable package named `transformer`, so the
`transformer.*` modules are importable from anywhere in the environment.

## Configuration

Pipeline behavior is controlled by two YAML files:

- **`config/config.yaml`** — paths, dataset source, and language pair for
  each stage (default: `en` → `fr`, `opus_books`).
- **`config/params.yaml`** — model/training hyperparameters:

  ```yaml
  model_config:
    seq_len: 512
    batch_size: 32
    learning_rate: 1e-4
    num_epochs: 17
    d_model: 512
  ```

Edit these before training to change the dataset, language pair, sequence
length, or model size. `seq_len` and `d_model` from `params.yaml` are what
`ModelTrainer.get_model()` passes into `build_transformer`; everything
else (`N`, `h`, `dropout`, `d_ff`) currently uses the defaults defined in
`build_transformer`'s signature in `model.py`.

## Running the training pipeline

```bash
python -m transformer.pipline.training_pipeline
```

This runs all four stages end-to-end and writes:

- Raw dataset → `artifacts/opus_books`
- Tokenizers → `artifacts/data_transformation/tokenizer_{en,fr}.json`
- Train/val splits → `artifacts/data/{train,val}`
- Checkpoints → `artifacts/model_trainer/model.pt` (overwritten each epoch)
- TensorBoard logs → `artifacts/model_trainer/logs`
- Timestamped run logs → `logs/`

Monitor training with:

```bash
tensorboard --logdir artifacts/model_trainer/logs
```

### Known issue: data path mismatch

As currently written, **data validation and data transformation don't read
from the path data ingestion writes to.** Data ingestion saves to
`artifacts/opus_books` (`config.data_ingestion.save_dir`), but
`DataValidation.validate_data()` and
`DataTransformation.initiate_data_transformation()` both call:

```python
load_dataset('json', data_files=f'opus_books_{lang_src}_{lang_tgt}.json', split='train')
```

— a hardcoded filename in the current working directory, ignoring
`config.data_path`/`config.dataset_path`. Running the pipeline fresh (with
only ingestion having run) will fail at the validation step unless a file
named `opus_books_en_fr.json` exists in the directory you launch the
command from.

**Workaround** until this is fixed upstream: after data ingestion
completes, copy or symlink the ingested file to the expected name:

```bash
cp artifacts/opus_books opus_books_en_fr.json
```

(substitute `en_fr` for your configured `lang_src`/`lang_tgt` if you
change the language pair).

## Running the web UI

The Flask app in `app.py` serves a DeepL-style translator backed by a
trained checkpoint, calling into `translate.py` for inference.

1. Make sure you have a trained checkpoint and tokenizers available. By
   default `translate.py` is configured to load:
   - Checkpoint: `weightsv5_final_seq_len_update_required/tmodel_14.pt`
   - Tokenizers: `tokenizer_en.json`, `tokenizer_fr.json` (repo root)
   - `SEQ_LEN = 350`, `D_MODEL = 512` — **`SEQ_LEN` must match the
     `seq_len` the checkpoint was trained with**, since the positional
     encoding buffers are sized to it at build time (see
     [Architecture](#architecture)). The checkpoint currently shipped was
     trained at `seq_len=350`; a newer `seq_len=512` version is still
     training. Once that's ready, update `SEQ_LEN`, `CHECKPOINT_PATH`, and
     the tokenizer paths at the top of `translate.py`.

2. Start the server:

   ```bash
   python app.py
   ```

   The model and tokenizers are loaded once at startup (`warm_up()`), so
   the first request isn't slow.

3. Open `http://127.0.0.1:5000` in a browser, type English text in the
   left pane, and the French translation appears in the right pane
   (auto-translates as you type, with a short debounce). The page calls a
   `POST /api/translate` JSON endpoint (`{"text": "..."}` →
   `{"translation": "..."}` or `{"error": "..."}`).

> **Note:** `app.py` runs Flask's built-in development server
> (`debug=True`), which is not intended for production use and has no
> authentication. Don't expose it directly on a public network — put it
> behind a proper WSGI server (e.g. gunicorn, included in
> `requirements.txt`) and a reverse proxy if you need to serve it beyond
> local use.

## Translating from the command line

`translate.py` can also be used directly, without the web UI:

```bash
python translate.py "Hello, how are you today?"
```

This loads the same checkpoint/tokenizers as the web app and prints the
greedy-decoded French translation.

## Logs and artifacts

- `logs/` — timestamped log files from every pipeline run
  (`transformer/logger`).
- `artifacts/` — everything the pipeline produces (raw dataset,
  tokenizers, train/val splits, checkpoints, TensorBoard logs). Safe to
  delete to force a clean re-run (ingestion is skipped only if
  `artifacts/opus_books` already exists; the other stages always
  re-run).
- `runs/` — TensorBoard logs from the legacy `train_*.py` scripts (not the
  pipeline, which logs to `artifacts/model_trainer/logs` instead).

## Citation

```
@misc{vaswani2023attentionneed,
      title={Attention Is All You Need}, 
      author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
      year={2023},
      eprint={1706.03762},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1706.03762}, 
}
```
