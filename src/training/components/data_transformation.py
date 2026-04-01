import os
import sys 
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from sklearn.pipeline import Pipeline
from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.common import save_object

class DataTransformation:
    try:
        def __init__(self,train_path,test_path,target_column,save_location_train_arr,save_location_test_arr,preprocessor_path,encoder_path):
            self.train_path = train_path
            self.test_path = test_path
            self.target_column = target_column
            self.save_location_train_arr = save_location_train_arr
            self.save_location_test_arr = save_location_test_arr
            self.preprocessor_path = preprocessor_path
            self.encoder_path = encoder_path

        def initiate_data_transformation(self):
            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)
            logger.info("Datasets Read Successfully")

            X_train = train_df.drop(self.target_column,axis=1)
            X_test = test_df.drop(self.target_column,axis=1)
            y_train = train_df[self.target_column]
            y_test = test_df[self.target_column]
            logger.info("DataSet Splitted Successfully")
            label_encoder = LabelEncoder()

            if y_train.dtype == "object":
                y_train = label_encoder.fit_transform(y_train)
                y_test = label_encoder.transform(y_test)
            logger.info("Target Values Handled Successfully")
            num_cols = X_train.select_dtypes(exclude = ['object']).columns.tolist()
            cat_cols = X_train.select_dtypes(include = ['object']).columns.tolist()
            logger.info("Numerical and categorical columns separated")
            num_pipeline = Pipeline(
                    steps=[
                        ("imputer",SimpleImputer(strategy='median')),
                        ("scaler",StandardScaler())
                    ]
                )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("encoder",OneHotEncoder(handle_unknown='ignore')),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer([
                    ("num_pipeline",num_pipeline,num_cols),
                    ("cat_pipeline",cat_pipeline,cat_cols)
                ])
            
            train_features_processed = preprocessor.fit_transform(X_train)
            test_features_processed = preprocessor.transform(X_test)
            logger.info("Preprocessed train and test")
            
            train_arr = np.c_[
                    train_features_processed,np.array(y_train)
                ]

            test_arr = np.c_[
                test_features_processed,np.array(y_test)
            ]

            os.makedirs(os.path.dirname(self.save_location_train_arr), exist_ok=True)
            os.makedirs(os.path.dirname(self.save_location_test_arr), exist_ok=True)
            os.makedirs(os.path.dirname(self.preprocessor_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.encoder_path), exist_ok=True)

            np.save(self.save_location_train_arr, train_arr)
            np.save(self.save_location_test_arr, test_arr)
            logger.info("Train and test array saved successfully")
            save_object(self.preprocessor_path, preprocessor)
            save_object(self.encoder_path, label_encoder)
            logger.info("Preprocessor and encoder saved")
            return train_arr,test_arr
        
    except Exception as e:
        raise CustomException(e,sys)