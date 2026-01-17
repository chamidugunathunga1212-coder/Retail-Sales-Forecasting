from dataclasses import dataclass

@dataclass
class DataValidationArtifact:
    report_file_path: str
    validated_train_file_path: str
    validated_test_file_path: str
    validation_status: bool