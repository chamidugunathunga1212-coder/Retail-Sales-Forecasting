from dataclasses import dataclass
import os
from datetime import datetime
from src.entity.config_entity.training_pipeline_config import TrainingPipelineConfig

@dataclass
class DataIngesionConfig:
    


    raw_data_dir: str = os.path.join(TrainingPipelineConfig.artifact_dir,"data_ingestion","feature_store")
    raw_data_path: str = os.path.join(raw_data_dir,"walmart_raw.csv") 

    train_data_path: str = os.path.join(TrainingPipelineConfig.artifact_dir,"data_ingestion","ingested","train.csv")
    test_data_path: str = os.path.join(TrainingPipelineConfig.artifact_dir,"data_ingestion","ingested","test.csv")

    test_size = 0.2


    

    
