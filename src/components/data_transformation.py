import pandas as pd
import numpy as np
import os
import sys

from src.entity.artifact_entity.data_transformation_artifact import DataTranformationArtifact
from src.entity.artifact_entity.data_validation_artifact import DataValidationArtifact
from src.entity.config_entity.data_transformation_config import DataTransformationConfig

from src.logging.logging import logging
from src.exception.exception import CustomerException

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

from scipy import sparse

from src.utils.main_utils import read_yaml,save_object,save_numpy_array

class DataTransformation:

    def __init__(self,data_transformation_config:DataTransformationConfig,data_validation_artifact:DataValidationArtifact):

        """
        YAML-driven, production-grade transformation.
        - Reads feature logic from config/feature_engineering.yaml
        - Ensures training-serving parity (same preprocessor for both)
        - Saves preprocessor + transformed arrays
        """

        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self.fe_config = read_yaml(self.data_transformation_config.feature_config_path)

            self.target_col = self.fe_config["target_column"]
            self.date_col = self.fe_config.get("date_column")

            self.cat_cols = self.fe_config.get("categorical_columns",[])
            self.num_cols = self.fe_config.get("numerical_columns",[])

            self.date_configs = self.fe_config.get("date_features",{"enabled": False, "features": []}) 
            self.target_configs = self.fe_config.get("target_transform",{"type": "none"})

            self.num_imputer_config = self.fe_config.get("numerical_imputer",{"strategy": "median"})
            self.cat_imputer_config = self.fe_config.get("categorical_imputer",{"strategy": "most_frequent"})

            self.scaling_config = self.fe_config.get("scaling",{"enabled": True, "method": "standard"})
            self.encodig_config = self.fe_config.get("encoding",{"method": "onehot", "handle_unknown": "ignore"})

            self.target_rules_config = self.fe_config.get("target_rules", {})


        except Exception as e:
            raise CustomerException(e,sys)


    def  _add_feature_engineering(self,df:pd.DataFrame)->pd.DataFrame:
        try:
            
            if not self.date_configs.get("enabled", False):
                return df
            
            if not self.date_col or self.date_col not in df.columns:
                # Don't hard fail, but log it (schema validation should catch this earlier)
                logging.warning("date_features enabled but date_column not found. Skipping time features.")
                return df
            
            df = df.copy()
            date = pd.to_datetime(df[self.date_col],errors="coerce")

            requested = set(self.date_configs.get("features", []))

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
        

    def _handle_negative_target(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Applies policy for negative target values.
            For this project we use: drop (recommended for forecasting + log1p).
            """

            try:
                
                if self.target_col not in df.columns:
                    return df
                
                policy = (self.target_rules_config.get("negative_target_policy") or "drop").lower()

                neg_mask = df[self.target_col].astype(float) < 0
                neg_count = int(neg_mask.sum())

                if neg_count == 0:
                    return df
                
                if policy == "drop":
                    logging.warning(f"Dropping {neg_count} rows with negative {self.target_col}")
                    return df.loc[~neg_mask].copy()
                
                if policy == "clip_to_zero":
                    logging.warning(f"Clipping {neg_count} negative {self.target_col} values to 0")
                    df = df.copy()
                    df.loc[neg_mask, self.target_col] = 0
                    return df
                
                if policy == "keep":
                    logging.warning(
                        f"Keeping {neg_count} negative {self.target_col} values. "
                        "Note: log1p is not suitable for negative targets."
                         )
                    return df


            except Exception as e:
                raise CustomerException(e,sys)


    def _apply_target_transform(self,y:pd.Series)-> np.ndarray:

        try:

            transformation_type = self.target_configs.get("type" or "none") 
            y = y.astype(float)

            if transformation_type == "log1p":
                return np.log1p(y.to_numpy())

            return y.to_numpy() 
        
        except Exception as e:
            raise CustomerException(e,sys)


    def _get_scaler(self):
        try:
            
            if not self.scaling_config.get("enabled", True):
                return None
            
            method = (self.scaling_config.get("method") or "standard").lower()

            if method == "standard":
                return StandardScaler()
            if method == "minmax":
                return MinMaxScaler()



        except Exception as e:
            raise CustomerException(e,sys)    


    def _get_preprocessor(self,final_numeric_cols,final_categorical_cols)->ColumnTransformer:
        try:
            
            num_imputer = SimpleImputer(strategy=self.num_imputer_config.get("strategy","median"))
            cat_imputer = SimpleImputer(strategy=self.cat_imputer_config.get("strategy","most_frequent"))

            scaler = self._get_scaler()

            if scaler is None:
                raise CustomerException("StanderScaler is Empty",sys)
            
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer",num_imputer),
                    ("scaler",scaler)
                ]
            )

            if (self.encodig_config.get("method") or "onehot").lower() != "onehot":
                raise ValueError("Only onehot encoding is supported in this pipeline")
            
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer",cat_imputer),
                    ("onehot", OneHotEncoder(handle_unknown=self.encodig_config.get("handle_unknown", "ignore")))
                ]
            )

            return ColumnTransformer(
                transformers=[
                    ("num",numerical_pipeline,final_numeric_cols),
                    ("cat",categorical_pipeline,final_categorical_cols)
                ],
                remainder="drop"
            )


        except Exception as e:
            raise CustomerException(e,sys)  




    def initiate_data_transformation(self):
        try:
            
            train_df = pd.read_csv(self.data_validation_artifact.validated_train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.validated_test_file_path)

            # Create time features (based on YAML)
            train_df = self._add_feature_engineering(train_df)
            test_df = self._add_feature_engineering(test_df)

            # Build final feature lists

            ## get the newly created date features 
            new_date_features = self.date_configs.get("features",[]) if self.date_configs.get("enabled",False) else []

            final_numeric_cols = self.num_cols + new_date_features
            final_categorical_cols = self.cat_cols

            # Basic safety checks
            for col in final_numeric_cols + final_categorical_cols + [self.target_col]:
                if col not in train_df.columns:
                    raise ValueError(f"Missing column in train data: {col}")
                if col not in test_df.columns:
                    raise ValueError(f"Missing column in test data: {col}")
                
            train_df = self._handle_negative_target(train_df)
            test_df = self._handle_negative_target(test_df)    
                

            ## split independent and dependent features

            y_train = self._apply_target_transform(train_df[self.target_col])
            y_test = self._apply_target_transform(test_df[self.target_col])

            X_train = train_df.drop(columns=[self.target_col],axis=1)
            X_test = test_df.drop(columns=[self.target_col],axis=1)

            print("X_train shape:", X_train.shape)
            print("y_train shape:", y_train.shape)
            print("final_numeric_cols:", len(final_numeric_cols), final_numeric_cols[:10])
            print("final_categorical_cols:", len(final_categorical_cols), final_categorical_cols[:10])

            # Fit preprocessor on train only
            preprocessor = self._get_preprocessor(final_numeric_cols=final_numeric_cols,final_categorical_cols=final_categorical_cols)

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            ## combine train and test as numpy array

            if sparse.issparse(X_train_transformed):
                X_train_transformed = X_train_transformed.toarray()
            if sparse.issparse(X_test_transformed):
                X_test_transformed = X_test_transformed.toarray()

            train_arr = np.hstack([X_train_transformed, y_train.reshape(-1, 1)])
            test_arr  = np.hstack([X_test_transformed,  y_test.reshape(-1, 1)])

            train_arr = np.c_[X_train_transformed,y_train]
            test_arr = np.c_[X_test_transformed,y_test]

            ## save artifacts
            save_object(file_path=self.data_transformation_config.preprocessor_path,obj=preprocessor)
            save_numpy_array(file_path=self.data_transformation_config.transformed_train_file_path,array=train_arr)
            save_numpy_array(file_path=self.data_transformation_config.transformed_test_file_path,array=test_arr)


            return DataTranformationArtifact(
                preprocessor_file_path=self.data_transformation_config.preprocessor_path,
                train_numpy_array_file_path=self.data_transformation_config.transformed_train_file_path,
                test_numpy_array_file_path=self.data_transformation_config.transformed_test_file_path
            )



        except Exception as e:
            raise CustomerException(e,sys)             

        