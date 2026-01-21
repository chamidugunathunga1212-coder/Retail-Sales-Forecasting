from src.entity.config_entity.data_ingestion_config import DataIngesionConfig
from src.components.data_ingestion import DataIngestion
from src.entity.artifact_entity.data_ingestion_artifact import DataIngestionArtifact

from src.entity.config_entity.data_validation_config import DataValidationConfig
from src.components.data_validation import DataValidation

from src.entity.config_entity.data_transformation_config import DataTransformationConfig
from src.entity.artifact_entity.data_transformation_artifact import DataTranformationArtifact
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer
from src.entity.config_entity.model_trainer_config import ModelTrainerConfig
from src.entity.artifact_entity.model_trainer_artifact import ModelTrainerArtifact

from src.utils.artifact_utils import promote_to_latest

from src.entity.config_entity.training_pipeline_config import TrainingPipelineConfig

from src.cloud.s3_syncer import S3Sync

if __name__ == "__main__":
    config_ingestion = DataIngesionConfig()
    ingestion = DataIngestion(config_ingestion)

    artifact_ingestion = ingestion.initiate_data_ingestion()

    config_validation = DataValidationConfig()
    validation = DataValidation(data_validation_config=config_validation,data_ingestion_artifact=artifact_ingestion)
    artifact_validation = validation.initiate_data_validation()

    config_transformation = DataTransformationConfig()
    transformation = DataTransformation(data_transformation_config=config_transformation,data_validation_artifact=artifact_validation)
    artifact_transformation = transformation.initiate_data_transformation()


    config_model_trainer_config = ModelTrainerConfig()
    model_trainer = ModelTrainer(model_trainer_config=config_model_trainer_config,data_transformation_artifact=artifact_transformation)
    model_trainer_artifact = model_trainer.initiate_model_trainer()
    print(model_trainer_artifact)

    promote_to_latest(src_path=artifact_transformation.preprocessor_file_path,latest_path="artifacts/latest/preprocessor/preprocessor.pkl")
    promote_to_latest(src_path=model_trainer_artifact.best_model_path,latest_path="artifacts/latest/model/best_model.pkl")


    s3_sync = S3Sync()
    TRAINING_BUCKET_NAME = "myretailsalesforecasting"

    aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifacts/{TrainingPipelineConfig.timestamp}"
    s3_sync.sync_folder_to_s3(folder = TrainingPipelineConfig.artifact_dir,aws_bucket_url=aws_bucket_url)

    # 2) upload ONLY latest (recommended for EC2 serving)
    aws_bucket_url_latest = f"s3://{TRAINING_BUCKET_NAME}/artifacts/latest"
    s3_sync.sync_folder_to_s3(folder="artifacts/latest",aws_bucket_url=aws_bucket_url_latest)

    print("Artifact folder saved sucessfully...")