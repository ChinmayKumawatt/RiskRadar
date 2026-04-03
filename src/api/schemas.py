from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from src.api.services.prediction_service import PredictionService


prediction_service = PredictionService()
metadata = prediction_service.get_metadata()

common_field_descriptions = {
    canonical_feature: (
        "Used by: "
        + ", ".join(
            f"{disease_name} ({original_feature})"
            for disease_name, original_feature in metadata["feature_aliases"][
                canonical_feature
            ].items()
        )
    )
    for canonical_feature in metadata["all_features"]
}


RiskInput = create_model(
    "RiskInput",
    __base__=BaseModel,
    __config__=ConfigDict(extra="forbid"),
    **{
        canonical_feature: (
            Any | None,
            Field(
                default=None,
                description=common_field_descriptions[canonical_feature],
            ),
        )
        for canonical_feature in metadata["all_features"]
    },
)


class PredictionResponse(BaseModel):
    disease: str
    input: dict[str, Any]
    prediction: Any
    prediction_label: Any
    risk_detected: bool
    risk_label: str
    risk_probability: float | None = None
    class_probabilities: dict[str, float] | None = None


class BulkPredictionResponse(BaseModel):
    predictions: dict[str, PredictionResponse]
    skipped: dict[str, str]


class MetadataResponse(BaseModel):
    supported_diseases: list[str]
    shared_features: dict[str, int]
    disease_features: dict[str, list[str]]
    feature_aliases: dict[str, dict[str, str]]
    all_features: list[str]
