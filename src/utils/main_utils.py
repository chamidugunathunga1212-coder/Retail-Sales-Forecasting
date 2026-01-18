import os
import json
import sys
import yaml
from typing import Dict,Any
import joblib
import numpy as np

from src.exception.exception import CustomerException
from src.logging.logging import logging

def read_yaml(file_path:str):
    try:
        with open(file_path,"r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise CustomerException(e,sys)


def _save_json(obj: Dict[str, Any], path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4)  
            
    except Exception as e:
        raise CustomerException(e,sys)   



def save_object(file_path:str,obj)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        joblib.dump(obj,file_path)
        logging.info(f"Object saved: {file_path}")


    except Exception as e:
        raise CustomerException(e,sys)     


def load_object(file_path:str):
    try:
        
        return joblib.load(file_path)


    except Exception as e:
        raise CustomerException(e,sys)
    


def save_numpy_array(file_path:str,array:np.ndarray)->None:
    try:

        
        os.makedirs(os.path.pardir(file_path),exist_ok=True)
        np.save(file_path,array)
        logging.info(f"Numpy array saved: {file_path}")


    except Exception as e:
        raise CustomerException(e,sys)    
