"""
train.py - Supervised Model Training & Cross-Validation Engine
Author: Ahmad Fathin Naufal (Universitas Gadjah Mada)
Role: AI Engineer

Evaluates three supervised classification architectures:
1. Decision Tree Classifier (Interpretable baseline)
2. Random Forest Classifier (Ensemble bagging)
3. XGBoost Classifier (Ensemble boosting)
Target: >80% Out-of-Sample Classification Accuracy
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
import logging

logger = logging.getLogger("model_trainer")


class ModelTrainer:
    """Trains, cross-validates, and serializes meteorological classification models."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models_cfg = config.get("models", {})
        self.model_dir = Path(config.get("paths", {}).get("model_dir", "outputs/models"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, Any] = {}
        self.cv_scores: Dict[str, float] = {}

    def initialize_models(self) -> Dict[str, Any]:
        """Instantiates models using tuned hyperparameter configurations."""
        dt_cfg = self.models_cfg.get("decision_tree", {})
        rf_cfg = self.models_cfg.get("random_forest", {})
        xgb_cfg = self.models_cfg.get("xgboost", {})

        self.models = {
            "Decision_Tree": DecisionTreeClassifier(**dt_cfg),
            "Random_Forest": RandomForestClassifier(**rf_cfg),
            "XGBoost": XGBClassifier(**xgb_cfg)
        }
        return self.models

    def cross_validate_all(
        self, X_train: np.ndarray, y_train: np.ndarray, n_splits: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """Executes Stratified 5-Fold Cross Validation across all model candidates."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        results = {}

        if not self.models:
            self.initialize_models()

        logger.info(f"Executing {n_splits}-Fold Stratified Cross-Validation on {X_train.shape[0]} samples...")

        for name, model in self.models.items():
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
            mean_acc = float(np.mean(scores))
            std_acc = float(np.std(scores))
            self.cv_scores[name] = mean_acc

            logger.info(f"Model: {name:<15} | 5-Fold CV Accuracy: {mean_acc * 100:.2f}% (+/- {std_acc * 100:.2f}%)")
            results[name] = {
                "cv_accuracy_mean": mean_acc,
                "cv_accuracy_std": std_acc
            }

        return results

    def train_and_persist(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Trains final models on full training set and serializes artifacts to disk."""
        if not self.models:
            self.initialize_models()

        trained_models = {}
        for name, model in self.models.items():
            logger.info(f"Fitting final {name} on {X_train.shape[0]} training observations...")
            model.fit(X_train, y_train)
            trained_models[name] = model

            # Serialize artifact
            model_path = self.model_dir / f"{name.lower()}.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Serialized {name} to {model_path}")

        return trained_models
