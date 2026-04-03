import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.utils.exception import CustomException
from src.utils.logger import logger
from src.utils.common import save_object,evaluate_models

class ModelTrainerArtifact:
    def __init__(self, model_path, best_model_name, best_score, best_threshold):
        self.model_path = model_path
        self.best_model_name = best_model_name
        self.best_score = best_score
        self.best_threshold = best_threshold

class ModelTrainer:
    def __init__(self,config):
        self.config = config

    def initiate_model_training(self):
        try:
            train_arr = np.load(self.config.train_arr_path)
            test_arr = np.load(self.config.test_arr_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "XGBoost": XGBClassifier()
            }

            params = {
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20]
                },
                "XGBoost": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 6]
                }
            }

            threshold_config = {
                "Random Forest": True,
                "XGBoost": True
            }

            models_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
                threshold_config=threshold_config
            )

            best_model_name = max(
                models_report,
                key=lambda x: models_report[x]["metrics"]["test_f1"]
            )

            best_model = models_report[best_model_name]["model"]

            best_score = models_report[best_model_name]["metrics"]["test_f1"]

            best_threshold = models_report[best_model_name]["best_threshold"]

            save_object(self.config.model_save_path, best_model)

            return ModelTrainerArtifact(
                model_path=self.config.model_save_path,
                best_model_name=best_model_name,
                best_score=best_score,
                best_threshold=best_threshold
            )

        except Exception as e:
            raise CustomException(e, sys)
    