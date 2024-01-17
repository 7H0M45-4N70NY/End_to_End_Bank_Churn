from Bank_Churn.constants import *
from Bank_Churn.utils.common import read_yaml,create_directories
from Bank_Churn.entity.config_entity import (DataIngestionConfig,DataValidationConfig,DataTransformationConfig,
                                             DataTrainerConfig,DataEvaluationConfig)

class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH,
                schema_filepath = SCHEMA_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self)-> DataIngestionConfig:
        config=self.config.data_ingestion
        create_directories([config.root_dir])
        
        data_ingestion_config=DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_file_dir=config.local_file_dir
        )
        return data_ingestion_config
    def get_data_validation_config(self)->DataValidationConfig:
        config=self.config.data_validation
        schema=self.schema.COLUMNS
        create_directories([config.root_dir])

        data_validation_config=DataValidationConfig(
            root_dir=config.root_dir,
            target=config.target,
            all_schema=schema,
            result=config.result
        )
        return data_validation_config
    def get_data_transformation_config(self)->DataTransformationConfig:
        config=self.config.data_transformation 
        target=self.schema.TARGET_COLUMNS
        create_directories([config.root_dir])
        data_transformation_config=DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path
        )
        return data_transformation_config
    def get_model_training_config(self)->DataTrainerConfig:
        config=self.config.data_training
        params=self.params.XGBClassifier
        schema=self.schema.TARGET_COLUMNS
        create_directories([config.root_dir])
        model_trainer_config=DataTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name=config.model_name,
            target_column=schema.name,
            n_estimators=params.n_estimators,
            subsample=params.subsample,
            max_depth=params.max_depth,
            learning_rate=params.learning_rate,
            colsample_bytree=params.colsample_bytree
        )
        return model_trainer_config
    
    def model_eval_config(self) ->DataEvaluationConfig:
        config=self.config.model_evaluation 
        schema=self.schema.TARGET_COLUMNS
        create_directories([config.root_dir])
        
        model_evaluation_config =DataEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            metric_file_name=config.metric_file_name,
            target_column=schema.name
        ) 
        return model_evaluation_config