from dataclasses import dataclass

@dataclass
class ModelTrainerArtifact:
    best_model_path: str
    best_model_name: str
    best_rmse_log: float
    best_rmse_original: float
    