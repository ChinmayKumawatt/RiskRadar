import os
import sys
import pandas as pd
import numpy as np

from src.utils.exception import CustomException
from src.utils.logger import logger

class DataValidationArtifact:
    def __init__(self, train_file_path, test_file_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path

class DataValidation:
    def __init__(self,config):
        self.config  = config

    @staticmethod
    def _normalize_feature_column(series):
        if series.dtype != "object":
            return pd.to_numeric(series, errors="coerce")

        normalized = series.astype(str).str.strip().str.lower()
        normalized = normalized.replace({"?": np.nan, "nan": np.nan})

        category_mapping = {
            "yes": 1,
            "no": 0,
            "good": 1,
            "poor": 0,
            "present": 1,
            "notpresent": 0,
            "abnormal": 1,
            "normal": 0,
            "ckd": 1,
            "notckd": 0,
        }

        mapped = normalized.map(lambda value: category_mapping.get(value, value))
        numeric = pd.to_numeric(mapped, errors="coerce")

        return numeric

    def initiate_data_validation(self):
        try:
            train_df = pd.read_csv(self.config.train_path)
            test_df = pd.read_csv(self.config.test_path)

            logger.info("Data Validation Initiated")

            missing_features = [col for col in self.config.selected_features if col not in train_df.columns]

            if len(missing_features) > 0:
                raise CustomException(f"Missing features in train data: {missing_features}", sys)
            
            missing_features_test = [col for col in self.config.selected_features if col not in test_df.columns]

            if len(missing_features_test) > 0:
                raise CustomException(f"Missing features in test data: {missing_features_test}", sys)
            
            logger.info("Missing Feature Checked")

            train_df = train_df.drop_duplicates()

            train_df = train_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

            train_df.replace('?', np.nan, inplace=True)

            for col in self.config.selected_features:
                if col != self.config.target_column:
                    train_df[col] = self._normalize_feature_column(train_df[col])

            logger.info("Numerical Columns fixed ")

            test_df = test_df.drop_duplicates()

            test_df = test_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

            test_df.replace('?', np.nan, inplace=True)

            for col in self.config.selected_features:
                if col != self.config.target_column:
                    test_df[col] = self._normalize_feature_column(test_df[col])

            if self.config.target_column not in train_df.columns:
                raise CustomException("Target column missing in train data", sys)
            
            if self.config.target_column not in test_df.columns:
                raise CustomException("Target column missing in test data", sys)
            
            logger.info("Target Column validated")

            if train_df[self.config.target_column].isnull().sum() > 0:
                raise CustomException("Null values found in target column (train)", sys)
            
            if test_df[self.config.target_column].isnull().sum() > 0:
                raise CustomException("Null values found in target column (test)", sys)
            
            if set(train_df.columns) != set(test_df.columns):
                raise CustomException("Train and test columns mismatch", sys)

            os.makedirs(os.path.dirname(self.config.save_location_train), exist_ok=True)

            os.makedirs(os.path.dirname(self.config.save_location_test), exist_ok=True)

            train_df.to_csv(self.config.save_location_train, index = False)

            test_df.to_csv(self.config.save_location_test, index = False)

            logger.info("Train and test datasets saved successfully")

            return DataValidationArtifact(
                train_file_path=self.config.save_location_train,
                test_file_path=self.config.save_location_test
            )

        except Exception as e:
            raise CustomException(e,sys)
