import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.logger import logger
from src.utils.exception import CustomException

class DataIngestionArtifact:
    def __init__(self, train_file_path, test_file_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path

class DataIngestion:
    def __init__(self,config):
        self.config = config

    def initiate_data_ingestion(self):
        try:
            logger.info("Data ingestion initiated")

            df = pd.read_csv(self.config.dataset_path)
            logger.info("Dataset Read")
      
            X = df[self.config.selected_features]
            y = df[self.config.target_column]
            
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

            train_df = pd.concat([X_train,y_train],axis=1)
            test_df = pd.concat([X_test,y_test],axis=1)
            logger.info(f"Shape of train set {train_df.shape}")
            logger.info(f"Shape of test set {test_df.shape}")

            os.makedirs(os.path.dirname(self.config.train_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.config.test_path), exist_ok=True)

            # train_path = os.path.join(self.save_location_train)
            # test_path = os.path.join(self.save_location_test)

            train_df.to_csv(self.config.train_path, index = False)
            test_df.to_csv(self.config.test_path, index = False)
            logger.info("Train and test datasets saved successfully")

            return DataIngestionArtifact(
                train_file_path=self.config.train_path,
                test_file_path=self.config.test_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)
