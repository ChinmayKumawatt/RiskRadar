import os


ARTIFACTS_DIR = "artifacts"


class DataIngestionConfig:
    def __init__(self):
        ingestion_dir = os.path.join(ARTIFACTS_DIR, "data_ingestion")

        self.train_path = os.path.join(ingestion_dir, "train.csv")
        self.test_path = os.path.join(ingestion_dir, "test.csv")
        self.selected_features = []
        self.target_column = None
        self.dataset_path = None


class DataValidationConfig:
    def __init__(self):
        validation_dir = os.path.join(ARTIFACTS_DIR, "data_validation")

        # will be filled later from ingestion artifact
        self.train_path = None
        self.test_path = None

        self.selected_features = []
        self.target_column = None

        self.save_location_train = os.path.join(validation_dir, "train.csv")
        self.save_location_test = os.path.join(validation_dir, "test.csv")


class DataTransformationConfig:
    def __init__(self):
        transformation_dir = os.path.join(ARTIFACTS_DIR, "data_transformation")

        # will be filled later from validation artifact
        self.train_path = None
        self.test_path = None

        self.target_column = None

        self.save_location_train_arr = os.path.join(transformation_dir, "train.npy")
        self.save_location_test_arr = os.path.join(transformation_dir, "test.npy")

        self.preprocessor_path = os.path.join(transformation_dir, "preprocessor.pkl")
        self.encoder_path = os.path.join(transformation_dir, "encoder.pkl")


class ModelTrainerConfig:
    def __init__(self):
        model_dir = os.path.join(ARTIFACTS_DIR, "model_trainer")

        # will be filled later from transformation artifact
        self.train_arr_path = None
        self.test_arr_path = None

        self.model_save_path = os.path.join(model_dir, "model.pkl")