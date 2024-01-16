from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url : str
    local_file_dir : Path
    
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    target : Path
    result: str
    all_schema:dict
    