import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def inverse_log1p(arr: np.ndarray) -> np.ndarray:
    return np.expm1(arr)