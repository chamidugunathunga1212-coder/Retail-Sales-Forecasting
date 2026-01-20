import os
import shutil
import sys

from src.exception.exception import CustomerException
from src.logging.logging import logging


def promote_to_latest(src_path:str,latest_path:str)-> None:

    """

    Copies an artifact from a timestamped run folder to a stable latest/ folder.
    This is a common production pattern for serving.

    """

    try:

        os.makedirs(os.path.dirname(latest_path), exist_ok=True)
        shutil.copy2(src_path, latest_path)
        logging.info(f"Promoted artifact to latest: {latest_path}")
        
    except Exception as e:
        raise CustomerException(e,sys)