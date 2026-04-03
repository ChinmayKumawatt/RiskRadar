import os


class InferenceConfig:
    def __init__(self, disease_name, artifacts_root="artifacts"):
        self.disease_name = disease_name.strip().lower()
        self.artifacts_root = artifacts_root
        self.disease_dir = os.path.join(self.artifacts_root, self.disease_name)
        self.model_path = os.path.join(self.disease_dir, "model", "model.pkl")
        self.preprocessor_path = os.path.join(
            self.disease_dir, "data_transformation", "preprocessor.pkl"
        )
        self.encoder_path = os.path.join(
            self.disease_dir, "data_transformation", "encoder.pkl"
        )

    def validate(self):
        required_paths = {
            "model": self.model_path,
            "preprocessor": self.preprocessor_path,
            "encoder": self.encoder_path,
        }

        missing = [
            f"{name}: {path}"
            for name, path in required_paths.items()
            if not os.path.exists(path)
        ]

        if missing:
            raise FileNotFoundError(
                "Missing inference artifacts for "
                f"'{self.disease_name}': {', '.join(missing)}"
            )

        return self
