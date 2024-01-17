import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from Bank_Churn.components.data_transformation import DataTransformation
from Bank_Churn.config.configuration import ConfigurationManager
from Bank_Churn import logger

class PredictionPipline:
    def __init__(self):
        self.model=joblib.load(Path("artifacts/data_training/model.joblib"))
    def predict(self,data):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            preprocessor=DataTransformation(config=data_transformation_config)
            transformed_data=preprocessor.run_prediction_transformation(data)
            reference=pd.read_csv("artifacts/data_transformation/test.csv")
            reference=reference.drop("Exited",axis=1)
            reference_columns=set(reference.columns)
            transformed_columns=set(transformed_data.columns)
            missing_columns = list(reference_columns - transformed_columns)  
            cols = {}
            for col in missing_columns:
                cols[col] = [0]
            extra_df=pd.DataFrame(cols)
            final_df=pd.concat([extra_df,transformed_data],axis=1)
            columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                     'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                     'Geography_France', 'Geography_Germany', 
                     'Geography_Spain', 'Gender_Female','Gender_Male']#To reorder based on the trained data order
            prediction=self.model.predict(final_df[columns])
            return prediction
        except Exception as e:
            raise e


class CustomData:
    def __init__(self,
                 CreditScore:float,
                 Age:float,
                 Tenure:float,
                 Balance:float,
                 NumOfProducts:float,
                 HasCrCard:bool,
                 IsActiveMember:bool,
                 EstimatedSalary:float,
                 Geography:str,
                 Gender:str):
        
        self.CreditScore=CreditScore
        self.Age=Age
        self.Tenure=Tenure
        self.Balance=Balance
        self.NumOfProducts=NumOfProducts
        self.HasCrCard=HasCrCard
        self.IsActiveMember = IsActiveMember
        self.EstimatedSalary = EstimatedSalary
        self.Geography = Geography
        self.Gender=Gender
            
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'CreditScore':[self.CreditScore],
                'Age':[self.Age],
                'Tenure':[self.Tenure],
                'Balance':[self.Balance],
                'NumOfProducts':[self.NumOfProducts],
                'HasCrCard':[self.HasCrCard],
                'IsActiveMember':[self.IsActiveMember],
                'EstimatedSalary':[self.EstimatedSalary],
                'Geography':[self.Geography],
                'Gender':[self.Gender]
                }
            df = pd.DataFrame(custom_data_input_dict)
            logger.info('Dataframe Gathered')
            return df
        except Exception as e:
            logger.info('Exception Occured in prediction pipeline')
            raise e
