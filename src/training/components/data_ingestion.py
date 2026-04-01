import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.logger import logger
from src.utils.exception import CustomException

class DataIngestion:
    def __init__(self,dataset_path,target_column,selected_features,save_location_train,save_location_test):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.selected_features = selected_features
        self.save_location_train = save_location_train
        self.save_location_test = save_location_test

    def initiate_data_ingestion(self):
        try:
            logger.info("Data ingestion initiated")

            df = pd.read_csv(self.dataset_path)
            logger.info("Dataset Read")
            df = df.drop_duplicates()
            df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            df.replace('?', np.nan, inplace=True)

            logger.info("Basic Validation done")
            X = df[self.selected_features]
            y = df[self.target_column]
            
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

            train_df = pd.concat([X_train,y_train],axis=1)
            test_df = pd.concat([X_test,y_test],axis=1)
            logger.info(f"Shape of train set {train_df.shape}")
            logger.info(f"Shape of test set {test_df.shape}")

            os.makedirs(os.path.dirname(self.save_location_train), exist_ok=True)
            os.makedirs(os.path.dirname(self.save_location_test), exist_ok=True)

            # train_path = os.path.join(self.save_location_train)
            # test_path = os.path.join(self.save_location_test)

            train_df.to_csv(self.save_location_train, index = False)
            test_df.to_csv(self.save_location_test, index = False)
            logger.info("Train and test datasets saved successfully")

            return self.save_location_train, self.save_location_test
        
        except Exception as e:
            raise CustomException(e,sys)
