from src.inference.components.predictor import Predictor
from src.inference.config.inference_config import InferenceConfig
from src.inference.pipeline.inference_pipeline import InferencePipeline


class PredictionService:
    def __init__(self, artifacts_root="artifacts"):
        self.artifacts_root = artifacts_root
        self.pipeline = InferencePipeline(artifacts_root=artifacts_root)
        self.predictors = self._load_predictors()
        self.positive_labels = {
            "heart": 0,
            "diabetes": 1,
            "ckd": "ckd",
            "hypertension": 1,
        }
        self.disease_features = {
            disease_name: predictor.expected_features
            for disease_name, predictor in self.predictors.items()
        }
        self.feature_aliases = self._build_feature_aliases()
        self.shared_features = self._compute_shared_features()
        self.all_features = sorted(
            self.feature_aliases
        )

    def _load_predictors(self):
        predictors = {}

        for disease_name in self.pipeline.get_supported_diseases():
            predictors[disease_name] = Predictor(
                InferenceConfig(
                    disease_name=disease_name,
                    artifacts_root=self.artifacts_root,
                )
            )

        return predictors

    @staticmethod
    def _canonicalize_feature_name(feature_name):
        return feature_name.strip().lower()

    def _build_feature_aliases(self):
        feature_aliases = {}

        for disease_name, features in self.disease_features.items():
            for feature in features:
                canonical_name = self._canonicalize_feature_name(feature)
                feature_aliases.setdefault(canonical_name, {})
                feature_aliases[canonical_name][disease_name] = feature

        return feature_aliases

    def _compute_shared_features(self):
        return {
            feature: len(disease_map)
            for feature, disease_map in sorted(self.feature_aliases.items())
            if len(disease_map) > 1
        }

    def get_metadata(self):
        return {
            "supported_diseases": sorted(self.predictors),
            "shared_features": self.shared_features,
            "disease_features": self.disease_features,
            "feature_aliases": self.feature_aliases,
            "all_features": self.all_features,
        }

    def get_disease_features(self, disease_name):
        normalized_name = disease_name.strip().lower()

        if normalized_name not in self.disease_features:
            supported = ", ".join(sorted(self.disease_features))
            raise ValueError(
                f"Unsupported disease '{disease_name}'. Supported diseases: {supported}"
            )

        return self.disease_features[normalized_name]

    def build_payload_for_disease(self, disease_name, request_payload):
        required_features = self.get_disease_features(disease_name)
        missing_features = []
        disease_payload = {}

        for feature in required_features:
            canonical_name = self._canonicalize_feature_name(feature)
            value = request_payload.get(canonical_name)

            if value is None:
                missing_features.append(canonical_name)
            else:
                disease_payload[feature] = value

        if missing_features:
            raise ValueError(
                f"Missing required features for '{disease_name}': {missing_features}"
            )

        return disease_payload

    def predict_for_disease(self, disease_name, request_payload):
        normalized_name = disease_name.strip().lower()
        disease_payload = self.build_payload_for_disease(
            normalized_name,
            request_payload,
        )
        result = self.predictors[normalized_name].predict(disease_payload)[0]
        return self._attach_risk_metadata(normalized_name, result)

    def predict_all(self, request_payload):
        predictions = {}
        skipped = {}

        for disease_name in sorted(self.predictors):
            try:
                predictions[disease_name] = self.predict_for_disease(
                    disease_name,
                    request_payload,
                )
            except ValueError as exc:
                skipped[disease_name] = str(exc)

        return {
            "predictions": predictions,
            "skipped": skipped,
        }

    def _attach_risk_metadata(self, disease_name, result):
        positive_label = self.positive_labels[disease_name]
        result["risk_detected"] = str(result["prediction_label"]) == str(positive_label)
        result["risk_label"] = (
            "higher_risk_pattern" if result["risk_detected"] else "lower_risk_pattern"
        )

        class_probabilities = result.get("class_probabilities") or {}
        positive_key = str(positive_label)
        result["risk_probability"] = class_probabilities.get(positive_key)

        return result
