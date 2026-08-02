import os

from transformer.configuration.configuration import ConfigurationManager
from transformer.components.data_ingestion import DataIngestion
from transformer.components.data_validation import DataValidation
from transformer.components.data_transformation import DataTransformation
from transformer.components.model_trainer import ModelTrainer
from transformer.logger import logging

class TrainingPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        config = ConfigurationManager()
        
        # Data Ingestion
        logging.info("Starting Data Ingestion")
        data_ingestion_config = config.get_data_ingestion_config()
        if not os.path.exists(data_ingestion_config.save_dir):
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_data()
        logging.info("Data Ingestion Completed")
        
        # Data Validation

        logging.info("Starting Data Validation")
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation_artifact = data_validation.validate_data()
        
        if not data_validation_artifact.validation_status:
            logging.warning(f"Data validation warning: {data_validation_artifact.message}")
            logging.warning("Proceeding with transformation, but some sequences may be truncated")
        
        logging.info(f"Max source length: {data_validation_artifact.max_src_length}")
        logging.info(f"Max target length: {data_validation_artifact.max_tgt_length}")
        logging.info("Data Validation Completed")
        
        # Data Transformation
        logging.info("Starting Data Transformation")
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data Transformation Completed")
        # Model Training
        logging.info("Starting Model Training")
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model Training Completed")
        
        return model_trainer_artifact

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(e)
        raise e
