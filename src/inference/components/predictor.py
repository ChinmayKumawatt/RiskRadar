import sys

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from src.utils.common import load_object
from src.utils.exception import CustomException
from src.utils.logger import logger


class Predictor:
    def __init__(self, config):
        self.config = config.validate()
        self.model = load_object(self.config.model_path)
        self.preprocessor = load_object(self.config.preprocessor_path)
        self.encoder = load_object(self.config.encoder_path)
        self.expected_features = self._get_expected_features()
        self.numeric_features, self.categorical_features = self._get_feature_groups()

    def _get_expected_features(self):
        feature_names = getattr(self.preprocessor, "feature_names_in_", None)

        if feature_names is None:
            raise ValueError(
                f"Unable to infer input schema for '{self.config.disease_name}'."
            )

        return list(feature_names)

    def _get_feature_groups(self):
        numeric_features = []
        categorical_features = []

        for transformer_name, _, columns in getattr(self.preprocessor, "transformers_", []):
            if transformer_name == "num_pipeline":
                numeric_features.extend(list(columns))
            elif transformer_name == "cat_pipeline":
                categorical_features.extend(list(columns))

        return numeric_features, categorical_features

    def _to_dataframe(self, payload):
        if isinstance(payload, pd.DataFrame):
            df = payload.copy()
        elif isinstance(payload, dict):
            df = pd.DataFrame([payload])
        elif isinstance(payload, list):
            df = pd.DataFrame(payload)
        else:
            raise TypeError(
                "Payload must be a dict, a list of dicts, or a pandas DataFrame."
            )

        missing_features = [
            feature for feature in self.expected_features if feature not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing required features for '{self.config.disease_name}': "
                f"{missing_features}"
            )

        df = df[self.expected_features].copy()

        for feature in self.numeric_features:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors="coerce")

        for feature in self.categorical_features:
            if feature in df.columns:
                df[feature] = (
                    df[feature]
                    .astype("string")
                    .str.strip()
                    .replace({"?": pd.NA, "nan": pd.NA, "<NA>": pd.NA})
                )

        return df

    def _decode_predictions(self, predictions):
        if hasattr(self.encoder, "classes_"):
            return self.encoder.inverse_transform(np.asarray(predictions).astype(int))

        return predictions

    def _get_decoded_classes(self):
        if hasattr(self.model, "classes_"):
            return self._decode_predictions(self.model.classes_)

        return None

    def predict(self, payload):
        try:
            input_df = self._to_dataframe(payload)
            transformed_features = self.preprocessor.transform(input_df)

            if issparse(transformed_features):
                transformed_features = transformed_features.toarray()

            raw_predictions = self.model.predict(transformed_features)
            decoded_predictions = self._decode_predictions(raw_predictions)
            decoded_classes = self._get_decoded_classes()

            probability_matrix = None
            if hasattr(self.model, "predict_proba"):
                probability_matrix = self.model.predict_proba(transformed_features)

            results = []
            records = input_df.to_dict(orient="records")

            for idx, record in enumerate(records):
                result = {
                    "disease": self.config.disease_name,
                    "input": self._serialize_record(record),
                    "prediction": self._to_python_scalar(raw_predictions[idx]),
                    "prediction_label": self._to_python_scalar(decoded_predictions[idx]),
                }

                if probability_matrix is not None and decoded_classes is not None:
                    result["class_probabilities"] = {
                        str(self._to_python_scalar(decoded_classes[class_idx])): round(
                            float(probability_matrix[idx][class_idx]), 6
                        )
                        for class_idx in range(len(decoded_classes))
                    }

                results.append(result)

            logger.info(
                "Inference completed for %s with %s record(s)",
                self.config.disease_name,
                len(results),
            )

            return results

        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def _to_python_scalar(value):
        if pd.isna(value):
            return None

        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, float) and value.is_integer():
            return int(value)

        return value

    def _serialize_record(self, record):
        return {
            key: self._to_python_scalar(value)
            for key, value in record.items()
        }
