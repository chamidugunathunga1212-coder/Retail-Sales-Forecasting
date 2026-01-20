from dataclasses import dataclass

@dataclass
class PredictionInputConfig:

    preprocessor_stable_path: str = "artifacts/latest/preprocessor/preprocessor.pkl" 
    model_stable_path: str = "artifacts/latest/model/best_model.pkl"
    feature_configure_file_path = "config/feature_engineering.yaml"