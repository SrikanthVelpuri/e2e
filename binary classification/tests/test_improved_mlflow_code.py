import pytest
from unittest.mock import patch, MagicMock
import mlflow
import xgboost
import pandas as pd
import numpy as np
from binary_classification.improved_mlflow_code import (
    get_git_info, create_experiment, load_data, create_model_artifacts,
    objective, train_model, evaluate_model, main
)

@pytest.fixture
def mock_mlflow():
    with patch('mlflow.start_run'), patch('mlflow.log_params'), patch('mlflow.log_metric'), patch('mlflow.log_figure'), patch('mlflow.log_artifact'), patch('mlflow.set_experiment'), patch('mlflow.create_experiment'), patch('mlflow.get_experiment_by_name'):
        yield

@pytest.fixture
def mock_xgboost():
    with patch('xgboost.XGBClassifier') as mock:
        yield mock

def test_get_git_info():
    with patch('git.Repo') as mock_repo:
        mock_repo.return_value.head.object.hexsha = 'dummy_sha'
        mock_repo.return_value.active_branch.name = 'dummy_branch'
        mock_repo.return_value.remotes.origin.url = 'dummy_url'
        git_info = get_git_info()
        assert git_info == {
            "git_commit": 'dummy_sha',
            "git_branch": 'dummy_branch',
            "git_repo": 'dummy_url'
        }

def test_create_experiment(mock_mlflow):
    experiment_id = create_experiment()
    assert experiment_id is not None

def test_load_data():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] > 0
    assert y_test.shape[0] > 0

def test_create_model_artifacts(mock_mlflow, mock_xgboost):
    model = MagicMock()
    X_test = pd.DataFrame(np.random.rand(10, 5), columns=[f'feature_{i}' for i in range(5)])
    y_test = np.random.randint(0, 2, size=10)
    create_model_artifacts(model, X_test, y_test, 'dummy_run_id')
    assert model.feature_importances_.shape[0] == 5

def test_objective(mock_mlflow, mock_xgboost):
    params = {
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0
    }
    result = objective(params)
    assert 'loss' in result
    assert 'status' in result

def test_train_model(mock_mlflow, mock_xgboost):
    best_params = {
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0
    }
    model = train_model(best_params)
    assert model is not None

def test_evaluate_model(mock_mlflow, mock_xgboost):
    model = MagicMock()
    X_test = pd.DataFrame(np.random.rand(10, 5), columns=[f'feature_{i}' for i in range(5)])
    y_test = np.random.randint(0, 2, size=10)
    metrics = evaluate_model(model, 'dummy_run_id')
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'roc_auc' in metrics

def test_main(mock_mlflow, mock_xgboost):
    with patch('binary_classification.improved_mlflow_code.load_data') as mock_load_data, patch('binary_classification.improved_mlflow_code.fmin') as mock_fmin:
        mock_load_data.return_value = (pd.DataFrame(np.random.rand(100, 5)), pd.DataFrame(np.random.rand(50, 5)), np.random.randint(0, 2, size=100), np.random.randint(0, 2, size=50))
        mock_fmin.return_value = {
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0
        }
        main()
