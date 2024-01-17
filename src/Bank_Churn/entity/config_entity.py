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
    
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path

@dataclass(frozen=True)
class DataTrainerConfig:
    root_dir:Path
    train_data_path:Path
    test_data_path:Path
    model_name:str
    target_column:str
    n_estimators:float
    subsample:float
    max_depth:float
    learning_rate:float
    colsample_bytree:float

@dataclass(frozen=True)
class DataEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file_name: Path
    target_column: str