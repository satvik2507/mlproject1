from src.MLproject.logger import logging  
from src.MLproject.exception import CustomException
from src.MLproject.components.data_ingestion import DataIngestion
import sys




if __name__ == "__main__":
  logging.info("Starting the ML Project")

  try:
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()

  except Exception as e:
    logging.info("Exception occurred")
    raise CustomException(e, sys)
