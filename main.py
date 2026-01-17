from src.entity.config_entity.data_ingestion_config import DataIngesionConfig
from src.components.data_ingestion import DataIngestion
from src.entity.artifact_entity.data_ingestion_artifact import DataIngestionArtifact

from src.entity.config_entity.data_validation_config import DataValidationConfig
from src.components.data_validation import DataValidation


from src.entity.config_entity.training_pipeline_config import TrainingPipelineConfig

if __name__ == "__main__":
    config_ingestion = DataIngesionConfig()
    ingestion = DataIngestion(config_ingestion)

    artifact_ingestion = ingestion.initiate_data_ingestion()

    config_validation = DataValidationConfig()
    validation = DataValidation(data_validation_config=config_validation,data_ingestion_artifact=artifact_ingestion)
    artifact = validation.initiate_data_validation()


    print(artifact)