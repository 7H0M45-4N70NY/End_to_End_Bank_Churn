from Bank_Churn.config.configuration import ConfigurationManager
from Bank_Churn.components.data_evaluation import ModelEvaluation
from Bank_Churn import logger

STAGE_NAME="Model Evaluation"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        config=ConfigurationManager()
        model_evaluation_config=config.model_eval_config()
        model_evaluation_config=ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.save_results()
    
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e 

    
        