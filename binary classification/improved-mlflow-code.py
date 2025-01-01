import shap
import xgboost
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
from mlflow.models import infer_signature
import mlflow.xgboost
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import logging
import pandas as pd
from urllib.parse import urlparse
import optuna
import time
import json
from datetime import datetime
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any
import os
import sys
import git

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set up OpenTelemetry tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(ConsoleSpanExporter())
)

# Constants
EXPERIMENT_NAME = "adult_income_prediction"
MODEL_NAME = "adult_income_classifier"
RANDOM_STATE = 42
TEST_SIZE = 0.33
CV_FOLDS = 5
MAX_EVALS = 50  # Increased from 20 for better optimization

def get_git_info() -> Dict[str, str]:
    """Get git repository information."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return {
            "git_commit": repo.head.object.hexsha,
            "git_branch": repo.active_branch.name,
            "git_repo": repo.remotes.origin.url
        }
    except Exception as e:
        logger.warning(f"Could not get git info: {e}")
        return {}

def create_experiment() -> str:
    """Create or get existing MLflow experiment with proper tags."""
    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment:
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(
                EXPERIMENT_NAME,
                tags={
                    "purpose": "income_prediction",
                    "dataset": "adult_income",
                    "created_at": datetime.now().isoformat()
                }
            )
        mlflow.set_experiment(EXPERIMENT_NAME)
        return experiment_id
    except Exception as e:
        logger.error(f"Error creating/getting experiment: {e}")
        raise

def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess the data with enhanced logging."""
    with tracer.start_as_current_span("load_data") as span:
        try:
            X, y = shap.datasets.adult()
            
            # Log dataset statistics
            dataset_stats = {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "class_distribution": np.bincount(y).tolist(),
                "feature_names": X.columns.tolist()
            }
            span.set_attribute("dataset_stats", json.dumps(dataset_stats))
            logger.info(f"Dataset statistics: {dataset_stats}")
            
            # Stratified split to maintain class distribution
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=TEST_SIZE, 
                random_state=RANDOM_STATE,
                stratify=y
            )
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

def create_model_artifacts(model, X_test, y_test, run_id: str) -> None:
    """Create and log model artifacts including feature importance and SHAP plots."""
    with tracer.start_as_current_span("create_artifacts") as span:
        try:
            # Feature importance plot
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance, x='importance', y='feature')
            plt.title('Feature Importance')
            mlflow.log_figure(plt.gcf(), "feature_importance.png")
            plt.close()
            
            # SHAP values and plots
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # Summary plot
            shap.summary_plot(shap_values, X_test, show=False)
            mlflow.log_figure(plt.gcf(), "shap_summary.png")
            plt.close()
            
            # Save feature importance as CSV
            feature_importance.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")
            
        except Exception as e:
            logger.error(f"Error creating model artifacts: {e}")
            span.set_attribute("error", str(e))
            raise

def objective(params: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced objective function for hyperparameter optimization."""
    with tracer.start_as_current_span("hyperopt_objective") as span:
        with mlflow.start_run(nested=True) as run:
            try:
                mlflow.log_params(params)
                
                # Use StratifiedKFold for better cross-validation
                skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                model = xgboost.XGBClassifier(**params)
                
                # Track multiple metrics
                cv_scores = {
                    'f1': [],
                    'accuracy': [],
                    'roc_auc': []
                }
                
                for train_idx, val_idx in skf.split(X_train, y_train):
                    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    model.fit(X_fold_train, y_fold_train)
                    y_pred = model.predict(X_fold_val)
                    y_prob = model.predict_proba(X_fold_val)[:, 1]
                    
                    cv_scores['f1'].append(f1_score(y_fold_val, y_pred))
                    cv_scores['accuracy'].append(accuracy_score(y_fold_val, y_pred))
                    cv_scores['roc_auc'].append(roc_auc_score(y_fold_val, y_prob))
                
                # Log mean and std of each metric
                for metric, scores in cv_scores.items():
                    mlflow.log_metric(f"cv_mean_{metric}", np.mean(scores))
                    mlflow.log_metric(f"cv_std_{metric}", np.std(scores))
                
                mean_f1 = np.mean(cv_scores['f1'])
                span.set_attribute("mean_f1_score", mean_f1)
                
                return {'loss': -mean_f1, 'status': STATUS_OK}
            except Exception as e:
                logger.error(f"Error in objective function: {e}")
                span.set_attribute("error", str(e))
                return {'loss': float('inf'), 'status': STATUS_OK}

def train_model(best_params: Dict[str, Any]) -> xgboost.XGBClassifier:
    """Enhanced model training with early stopping and learning curves."""
    with tracer.start_as_current_span("train_model") as span:
        start_time = time.time()
        try:
            # Add early stopping parameters
            early_stopping_params = {
                'early_stopping_rounds': 10,
                'eval_metric': ['logloss', 'error', 'auc'],
                'use_label_encoder': False
            }
            model_params = {**best_params, **early_stopping_params}
            
            model = xgboost.XGBClassifier(**model_params)
            
            # Train with evaluation set for early stopping
            eval_set = [(X_train, y_train), (X_test, y_test)]
            model.fit(
                X_train, 
                y_train,
                eval_set=eval_set,
                verbose=True
            )
            
            # Log training time and evaluation metrics
            training_time = time.time() - start_time
            mlflow.log_metric("training_time", training_time)
            span.set_attribute("training_time", training_time)
            
            # Log learning curves
            results = model.evals_result()
            for metric in results['validation_0']:
                plt.figure(figsize=(10, 6))
                plt.plot(results['validation_0'][metric], label='train')
                plt.plot(results['validation_1'][metric], label='test')
                plt.title(f'Learning Curve - {metric}')
                plt.xlabel('Iteration')
                plt.ylabel(metric)
                plt.legend()
                mlflow.log_figure(plt.gcf(), f"learning_curve_{metric}.png")
                plt.close()
            
            return model
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            span.set_attribute("error", str(e))
            raise

def evaluate_model(model: xgboost.XGBClassifier, run_id: str) -> Dict[str, float]:
    """Enhanced model evaluation with additional metrics and visualizations."""
    with tracer.start_as_current_span("evaluate_model") as span:
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            
            # Calculate comprehensive metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob[:, 1])
            }
            
            # Log detailed metrics and artifacts
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                span.set_attribute(f"metric_{metric_name}", metric_value)
            
            # Create and log model artifacts
            create_model_artifacts(model, X_test, y_test, run_id)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            span.set_attribute("error", str(e))
            raise

def main():
    """Enhanced main execution function with comprehensive MLflow tracking."""
    with tracer.start_as_current_span("main"):
        try:
            # Create or get experiment
            experiment_id = create_experiment()
            
            # Load data
            global X_train, X_test, y_train, y_test
            X_train, X_test, y_train, y_test = load_data()
            
            # Define hyperparameter search space
            space = {
                'max_depth': hp.choice('max_depth', range(3, 12)),
                'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.3)),
                'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400, 500, 600]),
                'min_child_weight': hp.choice('min_child_weight', range(1, 8)),
                'subsample': hp.uniform('subsample', 0.6, 1.0),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
                'gamma': hp.uniform('gamma', 0, 0.5)
            }
            
            with mlflow.start_run(experiment_id=experiment_id) as run:
                # Log git info and system info
                git_info = get_git_info()
                mlflow.set_tags(git_info)
                mlflow.set_tags({
                    "platform": sys.platform,
                    "python_version": sys.version,
                    "mlflow_version": mlflow.__version__,
                    "xgboost_version": xgboost.__version__
                })
                
                # Hyperparameter optimization
                trials = Trials()
                best = fmin(
                    fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=MAX_EVALS,
                    trials=trials
                )
                
                # Convert hyperopt results to actual parameter values
                best_params = {
                    'max_depth': best['max_depth'] + 3,
                    'learning_rate': best['learning_rate'],
                    'n_estimators': [100, 200, 300, 400, 500, 600][best['n_estimators']],
                    'min_child_weight': best['min_child_weight'] + 1,
                    'subsample': best['subsample'],
                    'colsample_bytree': best['colsample_bytree'],
                    'gamma': best['gamma'],
                    'objective': 'binary:logistic'
                }
                
                # Log best parameters
                mlflow.log_params(best_params)
                
                # Train and evaluate model
                model = train_model(best_params)
                metrics = evaluate_model(model, run.info.run_id)
                
                # Log model with signature and input example
                signature = infer_signature(X_train, model.predict(X_train))
                input_example = X_train.iloc[[0]]
                mlflow.xgboost.log_model(
                    model,
                    "model",
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=MODEL_NAME
                )
                
                # Log hyperopt trials history
                trials_dict = {
                    'trial_id': list(range(len(trials.trials))),
                    'loss': [-t['result']['loss'] for t in trials.trials],
                    'iteration': list(range(len(trials.trials)))
                }
                trials_df = pd.DataFrame(trials_dict)
                
                # Plot and log hyperparameter optimization history
                plt.figure(figsize=(10, 6))
                plt.plot(trials_df['iteration'], trials_df['loss'])
                plt.xlabel('Iteration')
                plt.ylabel('F1 Score')
                plt.title('Hyperparameter Optimization History')
                mlflow.log_figure(plt.gcf(), "hyperopt_history.png")
                plt.close()

                # Create and log evaluation dataset
                eval_data = X_test.copy()
                eval_data["label"] = y_test

                # Evaluate using MLflow's built-in evaluator
                model_uri = f"runs:/{run.info.run_id}/model"
                result = mlflow.evaluate(
                    model_uri,
                    eval_data,
                    targets="label",
                    model_type="classifier",
                    evaluators=["default"]
                )

                # Log final status and information
                logger.info(f"Best parameters: {best_params}")
                logger.info(f"Metrics: {metrics}")
                logger.info(f"MLflow Run ID: {run.info.run_id}")
                logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

        except Exception as e:
            logger.error(f"Error in main execution: {e}")
            raise

if __name__ == "__main__":
    main()