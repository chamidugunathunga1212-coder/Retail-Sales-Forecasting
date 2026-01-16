from src.entity.config_entity.data_ingestion_config import DataIngesionConfig
from src.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    config = DataIngesionConfig()
    ingestion = DataIngestion(config)

    artifact = ingestion.initiate_data_ingestion()
    print(artifact)