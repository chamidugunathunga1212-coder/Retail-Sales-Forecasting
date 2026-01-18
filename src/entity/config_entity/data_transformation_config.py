from dataclasses import dataclass
import os

from src.entity.config_entity.training_pipeline_config import TrainingPipelineConfig

class DataTransformationConfig:

    artifact_dir = TrainingPipelineConfig.artifact_dir
    transformation_dir = os.path.join(artifact_dir,"data_transformation")

    preprocessor_path = os.path.join(transformation_dir,"preprocessor","preprocessor.pkl")

    transformed_train_file_path = os.path.join(transformation_dir,"transformed","train")
    transformed_test_file_path = os.path.join(transformation_dir,"transformed","test")

    feature_config_path: str = os.path.join("config", "feature_engineering.yaml")

    