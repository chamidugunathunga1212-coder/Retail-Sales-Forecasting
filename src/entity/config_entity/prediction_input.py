from dataclasses import dataclass
import os

@dataclass
class PredictionInputConfig:

    preprocessor_stable_path: str = os.getenv(
        "PREPROCESSOR_PATH",
        "artifacts/latest/preprocessor/preprocessor.pkl"
    )
    model_stable_path: str = os.getenv(
        "MODEL_PATH",
        "artifacts/latest/model/best_model.pkl"
    )
    feature_configure_file_path: str = os.getenv(
        "FEATURE_CONFIG_PATH",
        "config/feature_engineering.yaml"
    )