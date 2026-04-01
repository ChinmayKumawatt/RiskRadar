import os
import sys
import pandas as pd
import numpy as np

from src.utils.exception import CustomException
from src.utils.logger import logger

class DataValidation:
    def __init__(self,train_path,test_path,selected_features,target_column,save_location_train,save_location_test):
        self.train_path = train_path
        self.test_path = test_path
        self.selected_features  = selected_features
        self.target_column  = target_column
        self.save_location_train = save_location_train
        self.save_location_test = save_location_test

    def initiate_data_validation(self):
        try:
            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)

            logger.info("Data Validation Initiated")

            missing_features = [col for col in self.selected_features if col not in train_df.columns]

            if len(missing_features) > 0:
                raise CustomException(f"Missing features in train data: {missing_features}", sys)
            
            missing_features_test = [col for col in self.selected_features if col not in test_df.columns]

            if len(missing_features_test) > 0:
                raise CustomException(f"Missing features in test data: {missing_features_test}", sys)
            
            logger.info("Missing Feature Checked")

            train_df = train_df.drop_duplicates()

            train_df = train_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

            train_df.replace('?', np.nan, inplace=True)

            for col in self.selected_features:
                if col != self.target_column:
                    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')

            logger.info("Numerical Columns fixed ")

            test_df = test_df.drop_duplicates()

            test_df = test_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

            test_df.replace('?', np.nan, inplace=True)

            for col in self.selected_features:
                if col != self.target_column:
                    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

            if self.target_column not in train_df.columns:
                raise CustomException("Target column missing in train data", sys)
            
            if self.target_column not in test_df.columns:
                raise CustomException("Target column missing in test data", sys)
            
            logger.info("Target Column validated")

            if train_df[self.target_column].isnull().sum() > 0:
                raise CustomException("Null values found in target column (train)", sys)
            
            if test_df[self.target_column].isnull().sum() > 0:
                raise CustomException("Null values found in target column (test)", sys)
            
            if set(train_df.columns) != set(test_df.columns):
                raise CustomException("Train and test columns mismatch", sys)

            os.makedirs(os.path.dirname(self.save_location_train), exist_ok=True)

            os.makedirs(os.path.dirname(self.save_location_test), exist_ok=True)

            train_df.to_csv(self.save_location_train, index = False)

            test_df.to_csv(self.save_location_test, index = False)

            logger.info("Train and test datasets saved successfully")

            return self.save_location_train, self.save_location_test

        except Exception as e:
            raise CustomException(e,sys)