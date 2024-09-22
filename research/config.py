from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10e-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": "opus_books",
        "lang_src": "en",
        "lang_tgt": "fr",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "run/tmodel"
    }

def get_weight_file_path(config, epoch):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return Path(".") / model_folder/ model_filename

def latest_weight_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weight_files = [Path(model_folder).glob(model_filename)]
    if len(weight_files) == 0:
        return None
    weight_files.sort()
    return weight_files[-1]
