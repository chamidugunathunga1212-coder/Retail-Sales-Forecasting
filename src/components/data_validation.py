import os
import sys
import pandas as pd
import numpy as np
import json
from typing import Dict,List, Any

from src.logging.logging import logging
from src.exception.exception import CustomerException

from src.entity.config_entity.data_validation_config import DataValidationConfig
from src.entity.artifact_entity.data_validation_artifact import DataValidationArtifact
from src.entity.artifact_entity.data_ingestion_artifact import DataIngestionArtifact

from src.utils.main_utils import read_yaml,_save_json


class DataValidation:

    def __init__(self,data_validation_config:DataValidationConfig,data_ingestion_artifact:DataIngestionArtifact):
        try:

            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self._schema = read_yaml(self.data_validation_config.schema_file_path)            

        except Exception as e:
            raise CustomerException(e,sys)
        

    def _missing_report(self,df:pd.DataFrame)->Dict[str,Any]:
        try:
            
            per_col = df.isna().mean().sort_values(ascending=False)
            total_cells = int(df.shape[0]*df.shape[1])
            total_missing = int(df.isna().sum().sum())
            overall_ratio = float(total_missing / total_cells) if total_cells else 0.0

            return {
                "total_missing": total_missing,
                "overall_missing_ratio": overall_ratio,
                "missing_ratio_per_column": {c: float(per_col[c]) for c in per_col.index},
            }

        except Exception as e:
            raise CustomerException(e,sys)

    
    def validate_columns(self,df:pd.DataFrame,required_cols:List[str],optional_cols:List[str])->Dict[str,Any]:
        try:
            
            df_cols = set(df.columns)
            req_cols = set(required_cols)
            opt_cols = set(optional_cols)

            missing_required = sorted(list(req_cols - df_cols))
            missing_optional = sorted(list(opt_cols - df_cols))
            extra = sorted(list(df_cols - (req_cols | opt_cols)))

            status = len(missing_required) == 0


        except Exception as e:
            raise CustomerException(e,sys)
        
        return {
            "status": status,
            "missing_required": missing_required,
            "missing_optional": missing_optional,
            "extra_columns": extra,
        }
    
    def validate_date_column(self,df:pd.DataFrame,date_col:str,max_nat_ratio: float)->Dict[str, Any]:
        try:
            
            if not date_col or date_col not in df.columns:
                return {
                "status": False,
                "reason": f"date_column '{date_col}' not found in dataframe",
                "nat_ratio": None,
                }
            

            parsed = pd.to_datetime(df[date_col], errors="coerce")
            nat_ratio = float(parsed.isna().mean())
            status = nat_ratio <= max_nat_ratio

            return {
            "status": status,
            "nat_ratio": nat_ratio,
            "max_nat_ratio_allowed": float(max_nat_ratio),
            }


        except Exception as e:
            raise CustomerException(e,sys)
        


    def _validate_target(self, df: pd.DataFrame, target_col: str, non_negative: bool) -> Dict[str, Any]:
        try:
            if target_col not in df.columns:
                return {
                "status": False,
                "reason": f"target_column '{target_col}' not found in dataframe",
                }

            if not non_negative:
                return {"status": True, "negative_count": 0}

            negative_count = int((df[target_col] < 0).sum())

            total_rows = len(df)
            negative_ratio = negative_count / total_rows if total_rows else 0.0

            return {
                "status": negative_count == 0,
                "negative_count": negative_count,
                "negative_ratio": float(negative_ratio)
            } 
        except Exception as e:
            raise CustomerException(e,sys)
        

    def _validate_duplicates(self, df: pd.DataFrame, allow_duplicates: bool) -> Dict[str, Any]:

        try:

            dup_count = int(df.duplicated().sum())
            status = True if allow_duplicates else (dup_count == 0)
        
            return {
                "status": status,
                "allow_duplicates": bool(allow_duplicates),
                "duplicate_count": dup_count,
            }  


        except Exception as e:
            raise CustomerException(e,sys)
        

    def _quality_rules_check(self, df: pd.DataFrame, rules: Dict[str, Any]) -> Dict[str, Any]:
        try:
            
            max_col_missing = float(rules.get("max_missing_ratio_per_column", 0.80))
            max_total_missing = float(rules.get("max_total_missing_ratio", 0.50))

            miss = self._missing_report(df)
            cols_exceeding = [c for c, r in miss["missing_ratio_per_column"].items() if r > max_col_missing]

            per_col_ok = len(cols_exceeding) == 0
            overall_ok = miss["overall_missing_ratio"] <= max_total_missing

            return {
                "status": bool(per_col_ok and overall_ok),
                "max_missing_ratio_per_column": max_col_missing,
                "max_total_missing_ratio": max_total_missing,
                "columns_exceeding_threshold": cols_exceeding,
                "missing_details": miss,
                "per_column_status": bool(per_col_ok),
                "overall_status": bool(overall_ok),
            }

        except Exception as e:
            raise CustomerException(e,sys)    
        

    def initiate_data_validation(self):
        try:
            
            target_col = self._schema["target_column"]["name"]
            date_col = self._schema.get("date_column",None)

            required_cols = self._schema.get("required_columns",[])
            optional_cols = self._schema.get("optional_columns",[])

            rules = self._schema.get("quality_rules",{})

            allow_duplicates = bool(rules.get("allow_duplicates",True))
            target_non_negative = bool(rules.get("target_non_negative",True))

            max_nat_ratio = float(rules.get("max_date_nat_ratio", 0.01))

            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            report: Dict[str, Any] = {
                "schema_file": self.data_validation_config.schema_file_path,
                "train_path": self.data_ingestion_artifact.train_file_path,
                "test_path": self.data_ingestion_artifact.test_file_path,
                "train": {},
                "test": {},
                "overall_status": True,
            }

            train_cols = self.validate_columns(df=train_df,required_cols=required_cols,optional_cols=optional_cols)
            train_dups = self._validate_duplicates(df=train_df,allow_duplicates=allow_duplicates)
            train_quality = self._quality_rules_check(df=train_df,rules=rules)
            train_target = self._validate_target(df=train_df,target_col=target_col,non_negative=target_non_negative)
            train_date = self.validate_date_column(df=train_df,date_col=date_col,max_nat_ratio=max_nat_ratio)

            report["train"] = {
                "columns": train_cols,
                "duplicates": train_dups,
                "quality": train_quality,
                "target": train_target,
                "date": train_date,
                "rows": int(train_df.shape[0]),
                "cols": int(train_df.shape[1]),
            }

            test_cols = self.validate_columns(df=test_df,required_cols=required_cols,optional_cols=optional_cols)
            test_dups = self._validate_duplicates(df=test_df,allow_duplicates=allow_duplicates)
            test_quality = self._quality_rules_check(df=test_df,rules=rules)
            test_target = self._validate_target(df=test_df,target_col=target_col,non_negative=target_non_negative)
            test_date = self.validate_date_column(df=test_df,date_col=date_col,max_nat_ratio=max_nat_ratio)

            report["test"] = {
                "columns": test_cols,
                "duplicates": test_dups,
                "quality": test_quality,
                "target": test_target,
                "date": test_date,
                "rows": int(test_df.shape[0]),
                "cols": int(test_df.shape[1]),
            }

            overall_status = all([
                train_cols["status"],
                test_cols["status"],
                train_dups["status"],
                test_dups["status"],
                train_quality["status"],
                test_quality["status"],
                train_target["status"],
                test_target["status"],
                train_date.get("status", True),
                test_date.get("status", True),
            ])

            report["overall_status"] = bool(overall_status)

            _save_json(report,self.data_validation_config.report_file_path)
            # Save validated copies (approved locations)
            os.makedirs(os.path.dirname(self.data_validation_config.validated_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_validation_config.validated_test_file_path), exist_ok=True)

            train_df.to_csv(self.data_validation_config.validated_train_file_path, index=False)
            test_df.to_csv(self.data_validation_config.validated_train_file_path, index=False)

            logging.info(f"Data Validation completed. Status: {overall_status}")

            return DataValidationArtifact(
                report_file_path=self.data_validation_config.report_file_path,
                validated_train_file_path=self.data_validation_config.validated_train_file_path,
                validated_test_file_path=self.data_validation_config.validated_test_file_path,
                validation_status=bool(overall_status),
            )

        except Exception as e:
            raise CustomerException(e,sys)    
   


