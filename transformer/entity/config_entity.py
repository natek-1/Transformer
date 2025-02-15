from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    datasource: str
    lang_src: str
    lang_tgt: str
    save_dir: Path
    
