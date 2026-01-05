from pathlib import Path

def get_config():
    return {
        "batch_size": 32,
        "num_epochs": 40,
        "lr": 10e-4,
        "seq_len": 512,
        "d_model": 512,
        "datasource": "opus_books",
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weightsv6",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "run/tmodel"
    }

def get_weights_file_path(config, epoch):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return Path(".") / model_folder/ model_filename

def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weight_files = list(Path(model_folder).glob(model_filename))
    if len(weight_files) == 0:
        return None
    weight_files.sort()
    return weight_files[-1]
