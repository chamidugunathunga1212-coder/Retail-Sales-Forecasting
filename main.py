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