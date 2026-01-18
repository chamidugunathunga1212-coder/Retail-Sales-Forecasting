from dataclasses import dataclass

@dataclass
class DataTranformationArtifact:

    preprocessor_file_path: str
    train_numpy_array_file_path: str
    test_numpy_array_file_path: str