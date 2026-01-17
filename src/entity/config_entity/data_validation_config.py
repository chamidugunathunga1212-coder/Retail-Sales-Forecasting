from dataclasses import dataclass
import os

from src.entity.config_entity.training_pipeline_config import TrainingPipelineConfig

@dataclass
class DataValidationConfig:
    schema_file_path: str = os.path.join("config","schema.yaml")

    validation_dir: str = os.path.join(TrainingPipelineConfig.artifact_dir,"data_validation")
    report_file_path: str = os.path.join(validation_dir,"validation_report.json")

    validated_train_file_path: str = os.path.join(validation_dir,"validated","train.csv")
    validated_test_file_path: str = os.path.join(validation_dir,"validated","test.csv")

    