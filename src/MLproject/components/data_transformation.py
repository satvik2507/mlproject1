import os
import sys
import string
import nltk
import pandas as pd

from dataclasses import dataclass

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

from src.MLproject.exception import CustomException
from src.MLproject.logger import logging
from src.MLproject.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.ps = PorterStemmer()


    def transform_text(self, text):

        try:
            text = str(text).lower()

            text = nltk.word_tokenize(text)

            y = []

            # Remove special characters
            for word in text:
                if word.isalnum():
                    y.append(word)

            text = y[:]
            y.clear()

            # Remove stopwords and punctuation
            for word in text:
                if (
                    word not in stopwords.words("english")
                    and word not in string.punctuation
                ):
                    y.append(word)

            text = y[:]
            y.clear()

            # Stemming
            for word in text:
                y.append(
                    self.ps.stem(word)
                )

            return " ".join(y)

        except Exception as e:
            raise CustomException(e, sys)


    def get_data_transformer_object(self):

        try:
            logging.info(
                "Creating Count Vectorizer"
            )

            vectorizer = CountVectorizer(
                max_features=3000
            )

            logging.info(
                "Count Vectorizer created successfully"
            )

            return vectorizer

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(
                "Read train and test data completed"
            )

            target_column_name = "Category"
            text_column_name = "Message"

            logging.info(
                "Starting text transformation"
            )

            train_df[text_column_name] = train_df[
                text_column_name
            ].apply(self.transform_text)

            test_df[text_column_name] = test_df[
                text_column_name
            ].apply(self.transform_text)

            logging.info(
                "Text transformation completed"
            )

            X_train = train_df[text_column_name]
            X_test = test_df[text_column_name]

            y_train = train_df[target_column_name]
            y_test = test_df[target_column_name]

            # Encode ham/spam
            label_encoder = LabelEncoder()

            y_train = label_encoder.fit_transform(
                y_train
            )

            y_test = label_encoder.transform(
                y_test
            )

            logging.info(
                "Target encoding completed"
            )

            preprocessing_obj = (
                self.get_data_transformer_object()
            )

            X_train = preprocessing_obj.fit_transform(
                X_train
            )

            X_test = preprocessing_obj.transform(
                X_test
            )

            logging.info(
                "Count Vectorization completed"
            )

            save_object(
                file_path=(
                    self.data_transformation_config
                    .preprocessor_obj_file_path
                ),
                obj=preprocessing_obj
            )

            logging.info(
                "Saved Count Vectorizer object"
            )

            return (
                X_train,
                X_test,
                y_train,
                y_test,
                self.data_transformation_config
                .preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)