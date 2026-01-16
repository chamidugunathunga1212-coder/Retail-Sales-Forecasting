from dataclasses import dataclass
import os
from datetime import datetime

@dataclass
class DataIngesionConfig:
    
    artifact_dir: str = os.path.join("artifacts",datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))

    raw_data_dir: str = os.path.join(artifact_dir,"data_ingestion","feature_store")
    raw_data_path: str = os.path.join(raw_data_dir,"walmart_raw.csv") 

    train_data_path: str = os.path.join(artifact_dir,"data_ingestion","ingested","train.csv")
    test_data_path: str = os.path.join(artifact_dir,"data_ingestion","ingested","test.csv")

    test_size = 0.2

    
