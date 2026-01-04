from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionArtifact:
    train_file_path: Path
    test_file_path: Path

@dataclass(frozen=True)
class DataTransformationArtifact:
    transformed_train_file_path: Path
    transformed_val_file_path: Path
    src_tokenizer_file_path: Path
    tgt_tokenizer_file_path: Path
