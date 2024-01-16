import os
import urllib.request as request
from Bank_Churn import logger
from Bank_Churn.entity.config_entity import DataIngestionConfig
from Bank_Churn.utils.common import get_size
from pathlib import Path
import pandas as pd


class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config
    def initiate_data_ingestion(self):
        logger.info("Data Ingestion Started")
        try :
            data=pd.read_csv(self.config.source_url)
            logger.info("Reading File")
            os.makedirs(os.path.dirname(os.path.join(self.config.local_file_dir)),exist_ok=True)
            data.to_csv(self.config.local_file_dir,index=False)
            logger.info(" i have saved the raw dataset in artifact folder")
        except Exception as e:
            logger.info("Data ingestion failed")
            raise e
            
            
            