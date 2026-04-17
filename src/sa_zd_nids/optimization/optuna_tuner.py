"""Hyperparameter optimization using Optuna for SA-ZD-NIDS."""
from __future__ import annotations

import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score
import joblib
import torch
from pathlib import Path
import json
from datetime import datetime

from sa_zd_nids.data.preprocessing import DataPreprocessor
from sa_zd_nids.models.classifier import KnownAttackClassifier
from sa_zd_nids.models.autoencoder import ZeroDayAutoencoder
from sa_zd_nids.evaluation.metrics import compute_classification_metrics


class OptunaTuner:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(self, config: Dict[str, Any], study_name: str = "sa-zd-nids"):
        self.config = config
        self.study_name = study_name
        self.preprocessor = None
        self.prepared_data = None
        
        # Optimize direction (maximize F1 macro)
        self.direction = "maximize"
        
    def prepare_data(self, input_path: str) -> None:
        """Prepare data for optimization."""
        self.preprocessor = DataPreprocessor(self.config)
        df = self.preprocessor.load_raw(input_path)
        self.prepared_data = self.preprocessor.fit_transform(df)
        
    def optimize_classifier(self, n_trials: int = 100) -> optuna.Study:
        """Optimize classifier hyperparameters."""
        
        def objective(trial: optuna.Trial) -> float:
            # Define search space
            model_type = trial.suggest_categorical("model_type", ["xgboost", "random_forest"])
            
            # XGBoost parameters
            if model_type == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10)
                }
            else:  # Random Forest
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 5, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
                }
            
            # Common parameters
            confidence_threshold = trial.suggest_float("confidence_threshold", 0.5, 0.9)
            
            # Update config
            trial_config = self.config.copy()
            trial_config["classifier"]["model_type"] = model_type
            trial_config["classifier"]["confidence_threshold"] = confidence_threshold
            
            if model_type == "xgboost":
                trial_config["classifier"]["xgboost"].update(params)
            else:
                trial_config["classifier"]["random_forest"].update(params)
            
            try:
                # Train classifier
                clf = KnownAttackClassifier(trial_config)
                clf_art = clf.fit(
                    self.prepared_data.X_train,
                    self.prepared_data.y_train,
                    self.prepared_data.X_val,
                    self.prepared_data.y_val
                )
                
                # Evaluate on validation set
                y_pred, _ = clf.predict(self.prepared_data.X_val)
                f1_macro = f1_score(self.prepared_data.y_val, y_pred, average="macro", zero_division=0)
                
                return f1_macro
                
            except Exception as e:
                print(f"Trial failed: {e}")
                return 0.0  # Return worst score for failed trials
        
        # Create study
        study = optuna.create_study(
            study_name=f"{self.study_name}_classifier",
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        return study
    
    def optimize_autoencoder(self, n_trials: int = 50) -> optuna.Study:
        """Optimize autoencoder hyperparameters."""
        
        def objective(trial: optuna.Trial) -> float:
            # Define search space
            hidden_dims = [
                trial.suggest_int("hidden_dim_1", 64, 256),
                trial.suggest_int("hidden_dim_2", 32, 128),
                trial.suggest_int("hidden_dim_3", 16, 64)
            ]
            
            # Ensure decreasing dimensions
            if not (hidden_dims[0] > hidden_dims[1] > hidden_dims[2]):
                return 0.0
            
            params = {
                "hidden_dims": hidden_dims,
                "epochs": trial.suggest_int("epochs", 10, 50),
                "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                "threshold_percentile": trial.suggest_int("threshold_percentile", 90, 99)
            }
            
            # Update config
            trial_config = self.config.copy()
            trial_config["anomaly"].update(params)
            
            try:
                # Get benign samples for autoencoder training
                benign_label = trial_config["data"]["benign_label"]
                train_normal = self.prepared_data.X_train[self.prepared_data.y_train == benign_label]
                val_normal = self.prepared_data.X_val[self.prepared_data.y_val == benign_label]
                
                if len(train_normal) == 0 or len(val_normal) == 0:
                    return 0.0
                
                # Train autoencoder
                ae = ZeroDayAutoencoder(trial_config)
                ae_art = ae.fit(train_normal, val_normal)
                
                # Evaluate reconstruction error on validation set
                val_errors = ae.reconstruction_error(val_normal)
                mean_error = np.mean(val_errors)
                
                # Lower mean reconstruction error is better
                return -mean_error  # Negative because we want to minimize
                
            except Exception as e:
                print(f"Trial failed: {e}")
                return 0.0
        
        # Create study
        study = optuna.create_study(
            study_name=f"{self.study_name}_autoencoder",
            direction="maximize",  # Since we return -mean_error
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        return study
    
    def optimize_joint(self, n_trials: int = 100) -> optuna.Study:
        """Joint optimization of classifier and autoencoder."""
        
        def objective(trial: optuna.Trial) -> float:
            # Classifier hyperparameters
            model_type = trial.suggest_categorical("model_type", ["xgboost", "random_forest"])
            
            if model_type == "xgboost":
                clf_params = {
                    "n_estimators": trial.suggest_int("clf_n_estimators", 50, 300),
                    "learning_rate": trial.suggest_float("clf_learning_rate", 0.01, 0.3, log=True),
                    "max_depth": trial.suggest_int("clf_max_depth", 3, 10),
                    "subsample": trial.suggest_float("clf_subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("clf_colsample_bytree", 0.6, 1.0),
                    "reg_lambda": trial.suggest_float("clf_reg_lambda", 0.1, 10.0, log=True)
                }
            else:
                clf_params = {
                    "n_estimators": trial.suggest_int("clf_n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("clf_max_depth", 5, 20),
                    "min_samples_split": trial.suggest_int("clf_min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("clf_min_samples_leaf", 1, 10)
                }
            
            # Autoencoder hyperparameters
            hidden_dims = [
                trial.suggest_int("ae_hidden_dim_1", 64, 256),
                trial.suggest_int("ae_hidden_dim_2", 32, 128),
                trial.suggest_int("ae_hidden_dim_3", 16, 64)
            ]
            
            if not (hidden_dims[0] > hidden_dims[1] > hidden_dims[2]):
                return 0.0
            
            ae_params = {
                "hidden_dims": hidden_dims,
                "epochs": trial.suggest_int("ae_epochs", 10, 30),
                "batch_size": trial.suggest_categorical("ae_batch_size", [128, 256, 512]),
                "learning_rate": trial.suggest_float("ae_learning_rate", 1e-4, 1e-2, log=True),
                "threshold_percentile": trial.suggest_int("ae_threshold_percentile", 90, 99)
            }
            
            # Common parameters
            confidence_threshold = trial.suggest_float("confidence_threshold", 0.5, 0.9)
            
            # Update config
            trial_config = self.config.copy()
            trial_config["classifier"]["model_type"] = model_type
            trial_config["classifier"]["confidence_threshold"] = confidence_threshold
            
            if model_type == "xgboost":
                trial_config["classifier"]["xgboost"].update(clf_params)
            else:
                trial_config["classifier"]["random_forest"].update(clf_params)
            
            trial_config["anomaly"].update(ae_params)
            
            try:
                # Train classifier
                clf = KnownAttackClassifier(trial_config)
                clf_art = clf.fit(
                    self.prepared_data.X_train,
                    self.prepared_data.y_train,
                    self.prepared_data.X_val,
                    self.prepared_data.y_val
                )
                
                # Train autoencoder
                benign_label = trial_config["data"]["benign_label"]
                train_normal = self.prepared_data.X_train[self.prepared_data.y_train == benign_label]
                val_normal = self.prepared_data.X_val[self.prepared_data.y_val == benign_label]
                
                if len(train_normal) == 0 or len(val_normal) == 0:
                    return 0.0
                
                ae = ZeroDayAutoencoder(trial_config)
                ae_art = ae.fit(train_normal, val_normal)
                
                # Evaluate hybrid performance
                # Simulate hybrid prediction on validation set
                y_pred_hybrid = []
                confidences = []
                
                for i in range(len(self.prepared_data.X_val)):
                    x = self.prepared_data.X_val[i:i+1]
                    pred_cls, conf = clf.predict(x)
                    pred_cls = pred_cls[0]
                    conf = conf[0]
                    
                    if conf < confidence_threshold:
                        flags, _ = ae.is_anomaly(x)
                        if flags[0]:
                            pred_cls = "ZERO_DAY"
                        else:
                            pred_cls = benign_label
                    
                    y_pred_hybrid.append(pred_cls)
                    confidences.append(conf)
                
                # Calculate metrics
                f1_macro = f1_score(self.prepared_data.y_val, y_pred_hybrid, average="macro", zero_division=0)
                
                return f1_macro
                
            except Exception as e:
                print(f"Trial failed: {e}")
                return 0.0
        
        # Create study
        study = optuna.create_study(
            study_name=f"{self.study_name}_joint",
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        return study
    
    def save_best_params(self, study: optuna.Study, output_path: str) -> None:
        """Save best hyperparameters to file."""
        best_params = study.best_params
        best_value = study.best_value
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results = {
            "study_name": study.study_name,
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": len(study.trials),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Best parameters saved to {output_path}")
        print(f"Best value: {best_value:.4f}")
        
        # Update config with best parameters
        self.update_config_with_best_params(best_params)
    
    def update_config_with_best_params(self, best_params: Dict[str, Any]) -> None:
        """Update config with best hyperparameters."""
        # Update classifier parameters
        if "model_type" in best_params:
            self.config["classifier"]["model_type"] = best_params["model_type"]
        
        if "confidence_threshold" in best_params:
            self.config["classifier"]["confidence_threshold"] = best_params["confidence_threshold"]
        
        # Update XGBoost parameters
        clf_xgb_params = {}
        for key, value in best_params.items():
            if key.startswith("clf_") and key != "clf_n_estimators":
                param_name = key.replace("clf_", "")
                if param_name in ["learning_rate", "max_depth", "subsample", "colsample_bytree", "reg_lambda"]:
                    clf_xgb_params[param_name] = value
        
        if clf_xgb_params:
            self.config["classifier"]["xgboost"].update(clf_xgb_params)
        
        # Update autoencoder parameters
        ae_params = {}
        for key, value in best_params.items():
            if key.startswith("ae_"):
                param_name = key.replace("ae_", "")
                ae_params[param_name] = value
        
        if ae_params:
            self.config["anomaly"].update(ae_params)
    
    def plot_optimization_history(self, study: optuna.Study, output_path: str) -> None:
        """Plot optimization history."""
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(output_path)
        print(f"Optimization history plot saved to {output_path}")
    
    def plot_param_importance(self, study: optuna.Study, output_path: str) -> None:
        """Plot parameter importance."""
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(output_path)
        print(f"Parameter importance plot saved to {output_path}")


def run_optimization(config_path: str, input_path: str, output_dir: str = "optimization_results"):
    """Run complete hyperparameter optimization."""
    import yaml
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize tuner
    tuner = OptunaTuner(config)
    
    # Prepare data
    print("Preparing data for optimization...")
    tuner.prepare_data(input_path)
    
    # Optimize classifier
    print("Optimizing classifier hyperparameters...")
    classifier_study = tuner.optimize_classifier(n_trials=100)
    tuner.save_best_params(classifier_study, f"{output_dir}/best_classifier_params.json")
    tuner.plot_optimization_history(classifier_study, f"{output_dir}/classifier_optimization_history.html")
    tuner.plot_param_importance(classifier_study, f"{output_dir}/classifier_param_importance.html")
    
    # Optimize autoencoder
    print("Optimizing autoencoder hyperparameters...")
    autoencoder_study = tuner.optimize_autoencoder(n_trials=50)
    tuner.save_best_params(autoencoder_study, f"{output_dir}/best_autoencoder_params.json")
    tuner.plot_optimization_history(autoencoder_study, f"{output_dir}/autoencoder_optimization_history.html")
    tuner.plot_param_importance(autoencoder_study, f"{output_dir}/autoencoder_param_importance.html")
    
    # Joint optimization
    print("Running joint optimization...")
    joint_study = tuner.optimize_joint(n_trials=100)
    tuner.save_best_params(joint_study, f"{output_dir}/best_joint_params.json")
    tuner.plot_optimization_history(joint_study, f"{output_dir}/joint_optimization_history.html")
    tuner.plot_param_importance(joint_study, f"{output_dir}/joint_param_importance.html")
    
    print("Optimization completed!")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    run_optimization(
        config_path="config.yaml",
        input_path="data/processed.csv",
        output_dir="optimization_results"
    )
