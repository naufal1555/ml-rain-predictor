"""
test_models.py - Unit Tests for ML Training & Evaluation
"""

import pytest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    X = np.random.randn(150, 13)
    # Synthetic target with strong correlation to feature 0 and 1
    y = ((X[:, 0] * 1.5 + X[:, 1] * 2.0 + np.random.randn(150) * 0.3) > 0).astype(int)
    return X, y


def test_model_initialization():
    config = {
        "models": {
            "decision_tree": {"max_depth": 5, "random_state": 42},
            "random_forest": {"n_estimators": 20, "max_depth": 5, "random_state": 42},
            "xgboost": {"n_estimators": 20, "max_depth": 4, "random_state": 42}
        },
        "paths": {"model_dir": "outputs/models"}
    }
    trainer = ModelTrainer(config)
    models = trainer.initialize_models()

    assert "Decision_Tree" in models
    assert "Random_Forest" in models
    assert "XGBoost" in models
    assert isinstance(models["Decision_Tree"], DecisionTreeClassifier)
    assert isinstance(models["Random_Forest"], RandomForestClassifier)
    assert isinstance(models["XGBoost"], XGBClassifier)


def test_cross_validation_execution(dummy_data):
    X, y = dummy_data
    config = {
        "models": {
            "decision_tree": {"max_depth": 3, "random_state": 42},
            "random_forest": {"n_estimators": 10, "max_depth": 3, "random_state": 42},
            "xgboost": {"n_estimators": 10, "max_depth": 3, "random_state": 42}
        },
        "paths": {"model_dir": "outputs/models"}
    }
    trainer = ModelTrainer(config)
    results = trainer.cross_validate_all(X, y, n_splits=3)

    assert "Decision_Tree" in results
    assert "Random_Forest" in results
    assert "XGBoost" in results
    for m, res in results.items():
        assert 0.0 <= res["cv_accuracy_mean"] <= 1.0
