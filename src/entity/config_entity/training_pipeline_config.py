from dataclasses import dataclass
import os
from datetime import datetime

@dataclass
class TrainingPipelineConfig:

    timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    artifact_dir: str = os.path.join("artifacts",timestamp)
