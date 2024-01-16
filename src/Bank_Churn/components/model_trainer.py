import pandas as pd
import os
from Bank_Churn import logger
from xgboost import XGBClassifier
import joblib
from Bank_Churn.entity.config_entity import DataTrainerConfig


class ModelTrainer:
    def __init__(self,config:DataTrainerConfig):
        self.config=config
    def train(self):
        train_data=pd.read_csv(self.config.train_data_path)
        test_data=pd.read_csv(self.config.test_data_path)
            
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]
        test_y = test_data[self.config.target_column]


        model = XGBClassifier(n_estimators=self.config.n_estimators,
                              subsample=self.config.subsample,max_depth=self.config.max_depth,
                              learning_rate=self.config.learning_rate,
                              colsample_bytree=self.config.colsample_bytree,random_state=42)
        model.fit(train_x, train_y)

        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))


        