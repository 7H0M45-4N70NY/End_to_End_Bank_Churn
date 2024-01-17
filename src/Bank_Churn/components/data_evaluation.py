import os 
import pandas as pd 
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from Bank_Churn.utils.common import save_json
import numpy as np 
import joblib
from Bank_Churn.entity.config_entity import DataEvaluationConfig
from pathlib import Path

class ModelEvaluation:
    def __init__(self,config:DataEvaluationConfig):
        self.config=config
    def eval_metrics(self,actual,preds):
        accuracy=accuracy_score(actual,preds)
        f1=f1_score(actual,preds)
        precision=precision_score(actual,preds)
        recall=recall_score(actual,preds)
        return accuracy,f1,precision,recall
    def save_results(self):
        test_data=pd.read_csv(self.config.test_data_path)
        model=joblib.load(self.config.model_path)
        test_x=test_data.drop([self.config.target_column],axis=1)
        test_y = test_data[[self.config.target_column]]
        
        predicted_qualities = model.predict(test_x)

        (accuracy,f1,precision,recall) = self.eval_metrics(test_y, predicted_qualities)   
        # Saving metrics as local
        scores = {"Accuracy":accuracy,"f1_score":f1,"Precision":precision,"Recall":recall}
        save_json(path=Path(self.config.metric_file_name), data=scores)

        