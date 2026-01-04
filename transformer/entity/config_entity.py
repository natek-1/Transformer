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
    
