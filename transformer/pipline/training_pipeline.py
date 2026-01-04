from transformer.configuration.configuration import ConfigurationManager
from transformer.components.data_ingestion import DataIngestion
from transformer.components.data_transformation import DataTransformation
from transformer.logger import logging

class TrainingPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        config = ConfigurationManager()
        
        # Data Ingestion
        logging.info("Starting Data Ingestion")
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_data()
        logging.info("Data Ingestion Completed")
        
        # Data Transformation
        logging.info("Starting Data Transformation")
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data Transformation Completed")
        
        return data_transformation_artifact

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(e)
        raise e
