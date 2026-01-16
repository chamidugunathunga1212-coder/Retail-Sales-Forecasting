import pandas as pd
import numpy as np
import os
import sys


from src.logging.logging import logging
from src.exception.exception import CustomerException

from src.entity.config_entity.data_ingestion_config import DataIngesionConfig
from src.entity.artifact_entity.data_ingestion_artifact import DataIngestionArtifact

from sklearn.model_selection import train_test_split

class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngesionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomerException(e,sys)
        

    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            
            logging.info("Starting Walmart data ingestion")

            train_df = pd.read_csv("data/train.csv")
            features_df = pd.read_csv("data/features.csv")
            stores_df = pd.read_csv("data/stores.csv")

            logging.info("Raw data files loaded")

            #  Merge datasets
            df = train_df.merge(
                features_df, on=["Store", "Date", "IsHoliday"], how="left"
            ).merge(
                stores_df, on="Store", how="left"
            )

            logging.info("Datasets merged successfully")

            #  Create feature store directory
            os.makedirs(self.data_ingestion_config.raw_data_dir, exist_ok=True)

            # Save merged raw data
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False)

            # Train-test split (time-aware later, random for now)
            train_set, test_set = train_test_split(
                df,
                test_size=self.data_ingestion_config.test_size,
                random_state=42
            )

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False)


            logging.info("Data ingestion completed successfully")

            return DataIngestionArtifact(
                raw_file_path=self.data_ingestion_config.raw_data_path,
                train_file_path=self.data_ingestion_config.train_data_path,
                test_file_path=self.data_ingestion_config.test_data_path
            )



        except Exception as e:
            raise CustomerException(e,sys)    