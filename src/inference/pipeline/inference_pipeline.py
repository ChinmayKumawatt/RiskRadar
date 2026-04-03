import os
import sys

from src.inference.components.predictor import Predictor
from src.inference.config.inference_config import InferenceConfig
from src.utils.exception import CustomException


class InferencePipeline:
    def __init__(self, artifacts_root="artifacts"):
        self.artifacts_root = artifacts_root

    def get_supported_diseases(self):
        if not os.path.exists(self.artifacts_root):
            return []

        diseases = []

        for entry in os.listdir(self.artifacts_root):
            disease_dir = os.path.join(self.artifacts_root, entry)
            model_path = os.path.join(disease_dir, "model", "model.pkl")
            preprocessor_path = os.path.join(
                disease_dir, "data_transformation", "preprocessor.pkl"
            )
            encoder_path = os.path.join(
                disease_dir, "data_transformation", "encoder.pkl"
            )

            if (
                os.path.isdir(disease_dir)
                and os.path.exists(model_path)
                and os.path.exists(preprocessor_path)
                and os.path.exists(encoder_path)
            ):
                diseases.append(entry)

        return sorted(diseases)

    def predict(self, disease_name, payload):
        try:
            predictor = Predictor(
                InferenceConfig(
                    disease_name=disease_name,
                    artifacts_root=self.artifacts_root,
                )
            )
            return predictor.predict(payload)

        except Exception as e:
            raise CustomException(e, sys)
