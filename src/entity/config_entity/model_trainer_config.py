from dataclasses import dataclass
import os
from src.entity.config_entity.training_pipeline_config import TrainingPipelineConfig

@dataclass
class ModelTrainerConfig:
    model_config_path: str = os.path.join("config","model.yaml")
    trainer_dir: str = os.path.join(TrainingPipelineConfig.artifact_dir,"model_trainer")
    best_model_path: str = os.path.join(trainer_dir,"model","best_model.pkl")