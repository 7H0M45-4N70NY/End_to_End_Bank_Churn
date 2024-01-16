from Bank_Churn.config.configuration import ConfigurationManager
from Bank_Churn.components.data_transformation import DataTransformation
from Bank_Churn import logger
from pathlib import Path

STAGE_NAME="Data Transformation Stage"

class DataTransformationTrainingPipline:
    def __init__(self):
        pass
    def main(self):
        try:
            with open(Path("artifacts/data_validation/result.txt"),'r') as f:
                status=f.read().split(" ")[-1]
            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                data_transformation.train_test_spliting()
            else:
                raise Exception("You data schema is not valid")

        except Exception as e:
            print(e)

                
                 