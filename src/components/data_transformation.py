import pandas as pd
import os
import sys

from src.entity.artifact_entity.data_transformation_artifact import DataTranformationArtifact
from src.entity.artifact_entity.data_validation_artifact import DataValidationArtifact
from src.entity.config_entity.data_transformation_config import DataTransformationConfig

from src.logging.logging import logging
from src.exception.exception import CustomerException

class DataTransformation:

    def __init__(self,data_transformation_config:DataTransformationConfig,data_validation_artifact:DataValidationArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise CustomerException(e,sys)


    def         

        