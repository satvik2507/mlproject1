import os
import sys

from src.MLproject.exception import CustomException
from src.MLproject.utils import load_object
from src.MLproject.text_preprocessing import transform_text


class PredictPipeline:

    def __init__(self):

        self.model_path = os.path.join(
            "artifacts",
            "model.pkl"
        )

        self.preprocessor_path = os.path.join(
            "artifacts",
            "preprocessor.pkl"
        )

    def predict(self, message):

        try:

            model = load_object(self.model_path)

            vectorizer = load_object(self.preprocessor_path)

            cleaned_message = transform_text(message)

            transformed_message = vectorizer.transform(
                [cleaned_message]
            )

            prediction = model.predict(
                transformed_message
            )

            return prediction[0]

        except Exception as e:
            raise CustomException(e, sys)