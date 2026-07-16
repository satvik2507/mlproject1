import sys

from src.MLproject.logger import logging
from src.MLproject.exception import CustomException

from src.MLproject.components.data_ingestion import DataIngestion
from src.MLproject.components.data_transformation import DataTransformation
from src.MLproject.components.model_trainer import ModelTrainer


if __name__ == "__main__":

    logging.info("Starting the ML Project")

    try:

        # Data Ingestion
        data_ingestion = DataIngestion()

        train_data_path, test_data_path = (
            data_ingestion.initiate_data_ingestion()
        )

        print("Data Ingestion Completed")

        # Data Transformation
        data_transformation = DataTransformation()

        (
            X_train,
            X_test,
            y_train,
            y_test,
            _
        ) = data_transformation.initiate_data_transformation(
            train_data_path,
            test_data_path
        )

        print("Data Transformation Completed")

        # Model Training
        model_trainer = ModelTrainer()

        report = model_trainer.initiate_model_trainer(
            X_train,
            X_test,
            y_train,
            y_test
        )

        print(report)

        print("Model Training Completed")

    except Exception as e:
        logging.info("Exception occurred")
        raise CustomException(e, sys)
