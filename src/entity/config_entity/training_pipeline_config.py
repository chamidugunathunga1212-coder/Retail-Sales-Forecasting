from dataclasses import dataclass
import os
from datetime import datetime

@dataclass
class TrainingPipelineConfig:

    artifact_dir: str = os.path.join("artifacts",datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
