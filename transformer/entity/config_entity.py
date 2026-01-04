from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    datasource: str
    lang_src: str
    lang_tgt: str
    save_dir: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_file: Path
    seq_len: int
    batch_size: int
    lang_src: str
    lang_tgt: str
    
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    data_path: Path
    lang_src: str
    lang_tgt: str
    seq_len: int
    
@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_name: str
    seq_len: int
    d_model: int
    lr: float
    num_epochs: int
    pre_load: bool
    batch_size: int
    lang_src: str
    lang_tgt: str
