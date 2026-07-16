import os
import sys

from dataclasses import dataclass

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.MLproject.exception import CustomException
from src.MLproject.logger import logging
from src.MLproject.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def evaluate_models(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        models
    ):

        try:
            report = {}

            for model_name, model in models.items():

                logging.info(f"Training {model_name}")

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                accuracy = accuracy_score(
                    y_test,
                    y_pred
                )

                precision = precision_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )

                recall = recall_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )

                f1 = f1_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )

                report[model_name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                }

                logging.info(
                    f"{model_name} - "
                    f"Accuracy: {accuracy:.4f}, "
                    f"Precision: {precision:.4f}, "
                    f"Recall: {recall:.4f}, "
                    f"F1 Score: {f1:.4f}"
                )

            return report

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_model_trainer(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        try:
            logging.info("Model training started")

            models = {

                "Multinomial Naive Bayes": MultinomialNB(),

                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    random_state=42
                ),
            }

            model_report = self.evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            logging.info(
                f"Model evaluation report: {model_report}"
            )

            best_model_name = max(
                model_report,
                key=lambda model_name:
                model_report[model_name]["f1_score"]
            )

            best_model_score = model_report[
                best_model_name
            ]["f1_score"]

            best_model = models[
                best_model_name
            ]

            logging.info(
                f"Best model found: {best_model_name}"
            )

            logging.info(
                f"Best model F1 score: {best_model_score:.4f}"
            )

            if best_model_score < 0.80:
                raise CustomException(
                    "No suitable model found with F1 score above 0.80",
                    sys
                )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(
                "Best model saved successfully"
            )

            return {
                "best_model": best_model_name,
                "f1_score": best_model_score,
                "metrics": model_report
            }

        except Exception as e:
            raise CustomException(e, sys)