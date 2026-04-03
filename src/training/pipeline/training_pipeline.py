import os
import sys

from src.training.components.data_ingestion import DataIngestion
from src.training.components.data_validation import DataValidation
from src.training.components.data_transformation import DataTransformation
from src.training.components.model_trainer import ModelTrainer

from src.training.config.training_config import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from src.utils.exception import CustomException
from src.utils.logger import logger


class TrainingPipeline:
    def __init__(self):
        pass

    def start(self):
        try:
            logger.info("Training Pipeline Started")

            # =========================
            # DISEASE CONFIGS
            # =========================
            disease_configs = {
                "heart": {
                    "features": ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach',
                                'exang', 'thal'],  # fill
                    "target": "target",
                    "dataset_path": "data/heart.csv"
                },
                "diabetes": {
                    "features": [ 'Glucose', 'BloodPressure',
                                'BMI', 'Age'],
                    "target": "Outcome",
                    "dataset_path": "data/diabetes.csv"
                },
                "ckd": {
                    "features": ['age', 'bp', 'htn', 'dm', 'appet'],
                    "target": "classification",
                    "dataset_path": "data/kidney.csv"
                },
                "hypertension": {
                    "features": ['male', 'age', 'cigsPerDay', 'BPMeds','totChol',  'BMI', 'heartRate'],
                    "target": "Risk",
                    "dataset_path": "data/hypertension.csv"
                }
            }

            results = {}

            # =========================
            # LOOP OVER DISEASES
            # =========================
            for disease_name, disease_info in disease_configs.items():

                logger.info(f"Starting pipeline for {disease_name}")

                # =========================
                # 1. DATA INGESTION
                # =========================
                ingestion_config = DataIngestionConfig()

                ingestion_config.selected_features = disease_info["features"]
                ingestion_config.target_column = disease_info["target"]
                ingestion_config.dataset_path = disease_info["dataset_path"]

                ingestion_config.train_path = os.path.join(
                    "artifacts", disease_name, "data_ingestion", "train.csv"
                )
                ingestion_config.test_path = os.path.join(
                    "artifacts", disease_name, "data_ingestion", "test.csv"
                )

                data_ingestion = DataIngestion(config=ingestion_config)
                ingestion_artifact = data_ingestion.initiate_data_ingestion()

                logger.info(f"{disease_name} - Data Ingestion Completed")

                # =========================
                # 2. DATA VALIDATION
                # =========================
                validation_config = DataValidationConfig()

                validation_config.train_path = ingestion_artifact.train_file_path
                validation_config.test_path = ingestion_artifact.test_file_path
                validation_config.selected_features = disease_info["features"]
                validation_config.target_column = disease_info["target"]

                validation_config.save_location_train = os.path.join(
                    "artifacts", disease_name, "data_validation", "train.csv"
                )
                validation_config.save_location_test = os.path.join(
                    "artifacts", disease_name, "data_validation", "test.csv"
                )

                data_validation = DataValidation(config=validation_config)
                validation_artifact = data_validation.initiate_data_validation()

                logger.info(f"{disease_name} - Data Validation Completed")

                # =========================
                # 3. DATA TRANSFORMATION
                # =========================
                transformation_config = DataTransformationConfig()

                transformation_config.train_path = validation_artifact.train_file_path
                transformation_config.test_path = validation_artifact.test_file_path
                transformation_config.target_column = disease_info["target"]

                transformation_config.save_location_train_arr = os.path.join(
                    "artifacts", disease_name, "data_transformation", "train.npy"
                )
                transformation_config.save_location_test_arr = os.path.join(
                    "artifacts", disease_name, "data_transformation", "test.npy"
                )
                transformation_config.preprocessor_path = os.path.join(
                    "artifacts", disease_name, "data_transformation", "preprocessor.pkl"
                )
                transformation_config.encoder_path = os.path.join(
                    "artifacts", disease_name, "data_transformation", "encoder.pkl"
                )

                data_transformation = DataTransformation(config=transformation_config)
                transformation_artifact = data_transformation.initiate_data_transformation()

                logger.info(f"{disease_name} - Data Transformation Completed")

                # =========================
                # 4. MODEL TRAINING
                # =========================
                model_trainer_config = ModelTrainerConfig()

                model_trainer_config.train_arr_path = transformation_artifact.train_arr_path
                model_trainer_config.test_arr_path = transformation_artifact.test_arr_path

                model_trainer_config.model_save_path = os.path.join(
                    "artifacts", disease_name, "model", "model.pkl"
                )

                model_trainer = ModelTrainer(config=model_trainer_config)
                model_trainer_artifact = model_trainer.initiate_model_training()

                logger.info(f"{disease_name} - Model Training Completed")

                # =========================
                # STORE RESULTS
                # =========================
                results[disease_name] = {
                    "model_path": model_trainer_artifact.model_path,
                    "best_model": model_trainer_artifact.best_model_name,
                    "score": model_trainer_artifact.best_score,
                    "threshold": model_trainer_artifact.best_threshold
                }

            logger.info("Training Pipeline Completed Successfully")

            return results

        except Exception as e:
            raise CustomException(e, sys)