from src.MLproject.logger import logging  
from src.MLproject.exception import CustomException
from src.MLproject.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.MLproject.components.data_ingestion import DataIngestionconfig
from src.MLproject.components.data_transformation import DataTransformation, DataTransformationConfig
import sys




if __name__ == "__main__":
  logging.info("Starting the ML Project")

  try:
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    
    
    

  except Exception as e:
    logging.info("Exception occurred")
    raise CustomException(e, sys)
