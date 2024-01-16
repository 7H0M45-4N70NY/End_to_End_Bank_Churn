import os
import numpy as np
import pandas as pd
from Bank_Churn import logger
from Bank_Churn.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from Bank_Churn.utils.common import save_bin
from imblearn.over_sampling import SMOTE



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    def run_transformation(self,df):
        cat_cols=["Geography","Gender"]
        drop_cols=['id',"CustomerId","Surname"]
        num_cols=['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
        df2=df.drop(columns=drop_cols)
        num_pipeline=Pipeline(
                    steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                    ])
        cat_pipeline=Pipeline(
                    steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OneHotEncoder(sparse_output=False,handle_unknown="ignore"))
                    ]
                )
        preprocessor_cat=ColumnTransformer([
                ('cat_pipeline',cat_pipeline,cat_cols)
                ])
        preprocessor_num=ColumnTransformer([
                ('num_pipeline',num_pipeline,num_cols)
                ])
        result=preprocessor_cat.fit_transform(df2)
        columns=preprocessor_cat.named_transformers_['cat_pipeline'].named_steps['ordinalencoder'].get_feature_names_out(cat_cols)
        cat=pd.DataFrame(result,columns=columns)
        result2=preprocessor_num.fit_transform(df2)
        num=pd.DataFrame(result2,columns=df2[num_cols].columns)
        transformed_data=pd.concat([num,cat],axis=1)
        return transformed_data

    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)
        target_column="Exited"
        y=data[['Exited']]
        X=data.drop(columns="Exited")
        transformed_x=self.run_transformation(X)
        data=pd.concat([transformed_x,y],axis=1)
        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)
        # train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)
        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)
        smote=SMOTE(random_state=1)
        X = train.drop(columns=target_column,axis=1)
        y=train[[target_column]]
        smote_X,smote_y=smote.fit_resample(X,y)
        train_new=pd.concat([smote_X,smote_y],axis=1)
        logger.info("oversampling to balance classes")
        logger.info("Oversampled train data shape")
        logger.info(train_new.shape)
        train_new.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        print(train.shape)
        print(test.shape)
    