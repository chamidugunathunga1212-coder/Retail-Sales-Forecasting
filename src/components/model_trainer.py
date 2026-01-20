import os
import sys
import numpy as np
import mlflow
import yaml

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from src.logging.logging import logging
from src.exception.exception import CustomerException
from src.utils.main_utils import load_numpy_array_data,save_object,read_yaml

from src.utils.model_eval_utils import rmse, inverse_log1p

from src.entity.config_entity.model_trainer_config import ModelTrainerConfig
from src.entity.artifact_entity.model_trainer_artifact import ModelTrainerArtifact
from src.entity.artifact_entity.data_transformation_artifact import DataTranformationArtifact

from sklearn.model_selection import RandomizedSearchCV


class ModelTrainer:

    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTranformationArtifact):
        try:
            
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self.models_configurations = read_yaml(self.model_trainer_config.model_config_path)


        except Exception as e:
            raise CustomerException(e,sys)
        

    def build_model(self,name:str,params:dict):
        try:
            
            name = name.lower()

            if name == "random_forest":
                return RandomForestRegressor(**params)
            if name == "gradient_boosting":
                return GradientBoostingRegressor(**params)
            if name == "xgboost":
                return XGBRegressor(**params)


        except Exception as e:
            raise CustomerException(e,sys) 


    def maybe_tune(self,model,tuning_config:dict,X_train,y_train):
        """
            return the fitted model, best params and tuning info

        """ 

        try:

            if not tuning_config or not bool(tuning_config.get("enabled", False)):
                model.fit(X_train,y_train)
                return model,{},{"tuned":False}
            
            method = (tuning_config.get("method") or "randomized").lower()

            if method != "randomized":
                raise ValueError(f"Only 'randomized' tuning supported now. Got: {method}")
            

            n_iter = int(tuning_config.get("n_iter",25))
            cv = int(tuning_config.get("cv",3))
            scoring = tuning_config.get("scoring","neg_root_mean_squared_error")
            param_dist = tuning_config.get("param_distributions",{}) or {}

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )

            search.fit(X_train,y_train)


            best_model = search.best_estimator_
            best_params = search.best_params_
            best_cv_score = float(search.best_score_)

            return best_model, best_params, {
            "tuned": True,
            "method": "randomized",
            "cv": cv,
            "n_iter": n_iter,
            "scoring": scoring,
            "best_cv_score": best_cv_score
        }


        except Exception as e:
            raise CustomerException(e,sys)
        

        

    def initiate_model_trainer(self):

        logging.info("Starting Model Training...")

        try:

            tracking_uri = self.models_configurations.get("mlflow",{}).get("tracking_uri","file:./mlruns")
            experiment_name = self.models_configurations.get("mlflow",{}).get("experiment_name","default")

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

            train_arr = load_numpy_array_data(self.data_transformation_artifact.train_numpy_array_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.test_numpy_array_file_path)

            ## speparate the independent and depenent 

            X_train,y_train_log = train_arr[:,:-1], train_arr[:,-1]
            X_test,y_test_log = test_arr[:,:-1], test_arr[:,-1]


            best_rmse_log = float("inf")
            best_model = None
            best_name = None
            best_rmse_original = float("inf")


            # --- train & log each model

            models_cfg = self.models_configurations.get("models", {})

            for model_name,model_configs in models_cfg.items():

                if not bool(model_configs.get("enabled",True)):
                    logging.info(f"Skipping disabled model: {model_name}")
                    continue
                    
                
                params = model_configs.get("params",{}) or {}
                tuning_config = model_configs.get("tuning",{}) or {}


                with mlflow.start_run(run_name=model_name):

                    logging.info(f"Building model: {model_name}")
                    model = self.build_model(model_name, params)

                    # Tune or train
                    fitted_model, best_params, tuning_info = self.maybe_tune(
                        model, tuning_config, X_train, y_train_log
                    )

                    # Predict
                    preds_log = fitted_model.predict(X_test)

                    # RMSE on log scale
                    score_log  = rmse(y_true=y_test_log,y_pred=preds_log)

                    # RMSE on original scale (interpretability)
                    y_test_original = inverse_log1p(y_test_log)
                    preds_original = inverse_log1p(preds_log)

                    scoring_original = rmse(y_test_original,preds_original)


                    # Log to MLflow
                    mlflow.log_params({f"base__{k}": v for k, v in params.items()})

                    if tuning_info.get("tuned"):

                        mlflow.log_params({f"best__{k}": v for k, v in best_params.items()})
                        mlflow.log_param("tuning_enabled", True)
                        mlflow.log_param("tuning_method", tuning_info.get("method"))
                        mlflow.log_param("tuning_cv", tuning_info.get("cv"))
                        mlflow.log_param("tuning_n_iter", tuning_info.get("n_iter"))
                        mlflow.log_param("tuning_scoring", tuning_info.get("scoring"))
                        mlflow.log_metric("best_cv_score", float(tuning_info.get("best_cv_score")))

                    else:
                        mlflow.log_param("tuning_enabled", False)

                    mlflow.log_metric("rmse_log", float(score_log))
                    mlflow.log_metric("rmse_original", float(scoring_original))

                    logging.info(
                        f"{model_name} tuned={tuning_info.get('tuned')} "
                        f"rmse_log={score_log:.4f} rmse_original={scoring_original:.4f}"
                    )

                    # Track best

                    if score_log < best_rmse_log:
                        best_rmse_log = score_log
                        best_rmse_original = scoring_original
                        best_model = fitted_model
                        best_name = model_name

            if best_model is None:
                raise ValueError("No models were trained. Check config/model.yaml enabled flags.")

            # Save best model

            save_object(file_path=self.model_trainer_config.best_model_path,obj=best_model)

            logging.info(f"Best model: {best_name} rmse_log={best_rmse_log:.4f}")   


            return ModelTrainerArtifact(
                best_model_path=self.model_trainer_config.best_model_path,
                best_model_name=best_name,
                best_rmse_log=float(best_rmse_log),
                best_rmse_original=float(best_rmse_original),
            ) 

        except Exception as e:
            raise CustomerException(e,sys)    
        