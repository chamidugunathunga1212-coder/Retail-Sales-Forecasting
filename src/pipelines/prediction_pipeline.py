import sys
import numpy as np
import pandas as pd

from src.exception.exception import CustomerException
from src.logging.logging import logging
from src.utils.main_utils import load_object,read_yaml

class PredictionPipeline:
    def __init__(self,preprocessor_path:str,model_path:str,feature_config_path:str):
        try:

            self.preprocessor = load_object(preprocessor_path)
            self.model = load_object(model_path)
            self.feature_config = read_yaml(feature_config_path)

            self.target_col = self.feature_config["target_column"]
            self.date_col = self.feature_config.get("date_column")

            self.date_config = self.feature_config.get("date_features",{"enabled": False, "features": []})
            


        except Exception as e:
            raise CustomerException(e,sys)
        

    def _add_time_features(self,df:pd.DataFrame)->pd.DataFrame:

        try:
            
            if not self.date_config.get("enabled",False):
                return df
            
            if not self.date_col or self.date_col not in df.columns:
                return df
            
            df = df.copy()

            date = pd.to_datetime(df[self.date_col],errors="coerce")
            requested = set(self.date_config.get("features", []))

            if "year" in requested:
                df["year"] = date.dt.year

            if "month" in requested:
                df["month"] = date.dt.month

            if "day" in requested:
                df["day"] = date.dt.day.astype("Int64")

            if "weekofyear" in requested:
                df["weekofyear"] = date.dt.isocalendar().week.astype("int64")

            if "dayofweek" in requested:
                df["dayofweek"] = date.dt.day_of_week

            if "is_month_start" in requested:
                df["is_month_start"] = date.dt.is_month_start.astype("int64")

            if "is_month_end" in requested:
                df["is_month_end"] = date.dt.is_month_end.astype("int64")

            df.drop(columns=[self.date_col], inplace=True)    

            return df 

        except Exception as e:
            raise CustomerException(e,sys)    
        

    def predict(self,input_df: pd.DataFrame)-> np.ndarray:
        """
        Returns predictions in ORIGINAL scale (Weekly_Sales),
        assuming model was trained on log1p(target).

        """  

        try:
            
            df = input_df.copy()

            
            # Create time features exactly like training
            df = self._add_time_features(df)

            X = self.preprocessor.transform(df)

            # Model outputs log-scale predictions
            preds_log = self.model.predict(X)

            # Convert back to original scale
            preds = np.expm1(preds_log)
            
            return preds


        except Exception as e:
            raise CustomerException(e,sys)  


        