"""
evaluate.py - Quantitative Model Benchmarking & Statistical Evaluation
Author: Ahmad Fathin Naufal (Universitas Gadjah Mada)
Role: AI Engineer

Evaluates Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, and Feature Importance.
Generates publication-grade figures saved to outputs/figures/.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import logging

logger = logging.getLogger("evaluator")

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


class ModelEvaluator:
    """Computes comprehensive classification metrics and renders diagnostic figures."""

    def __init__(self, figure_dir: Path, feature_names: List[str]):
        self.figure_dir = Path(figure_dir)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.feature_names = feature_names

    def evaluate_test_set(
        self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray
    ) -> pd.DataFrame:
        """Calculates out-of-sample metrics across all models."""
        metrics_list = []

        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_prob)

            metrics_list.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1_Score": round(f1, 4),
                "ROC_AUC": round(roc_auc, 4)
            })

            logger.info(
                f"Evaluation [{name:<15}] -> Accuracy: {acc*100:.2f}% | "
                f"F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | Recall: {rec:.4f}"
            )

        df_metrics = pd.DataFrame(metrics_list).sort_values(by="Accuracy", ascending=False)
        return df_metrics

    def plot_confusion_matrices(
        self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray
    ) -> Path:
        """Plots side-by-side normalized confusion matrices."""
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.5), dpi=300)
        if n_models == 1:
            axes = [axes]

        for ax, (name, model) in zip(axes, models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            sns.heatmap(
                cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"],
                cbar=False, ax=ax, annot_kws={"size": 12, "weight": "bold"}
            )
            ax.set_title(f"{name}\nAccuracy: {accuracy_score(y_test, y_pred)*100:.1f}%", fontsize=12, fontweight="bold")
            ax.set_xlabel("Predicted Rainfall Class")
            ax.set_ylabel("True Ground Truth")

        plt.tight_layout()
        out_path = self.figure_dir / "confusion_matrices.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved confusion matrices to {out_path}")
        return out_path

    def plot_roc_curves(
        self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray
    ) -> Path:
        """Plots comparative ROC curves with AUC annotations."""
        fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.50)")

        colors = {"Decision_Tree": "#E63946", "Random_Forest": "#2A9D8F", "XGBoost": "#1D3557"}

        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_score = roc_auc_score(y_test, y_prob)
                ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})", color=colors.get(name, "blue"), lw=2.2)

        ax.set_title("BMKG Rainfall Classification - ROC-AUC Comparative Curves", fontsize=13, fontweight="bold")
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
        ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#CBD5E1")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        plt.tight_layout()
        out_path = self.figure_dir / "roc_curves.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved ROC curves to {out_path}")
        return out_path

    def plot_feature_importance(self, models: Dict[str, Any]) -> Path:
        """Plots feature importance comparison for tree ensembles (Random Forest & XGBoost)."""
        tree_models = {k: v for k, v in models.items() if hasattr(v, "feature_importances_")}
        if not tree_models:
            return self.figure_dir

        fig, axes = plt.subplots(1, len(tree_models), figsize=(6.0 * len(tree_models), 6.0), dpi=300)
        if len(tree_models) == 1:
            axes = [axes]

        for ax, (name, model) in zip(axes, tree_models.items()):
            importances = model.feature_importances_
            df_imp = pd.DataFrame({"Feature": self.feature_names, "Importance": importances})
            df_imp = df_imp.sort_values(by="Importance", ascending=True)

            ax.barh(df_imp["Feature"], df_imp["Importance"], color="#008837" if name == "Random_Forest" else "#0B2545", alpha=0.85)
            ax.set_title(f"{name} - Feature Importance", fontsize=12, fontweight="bold")
            ax.set_xlabel("Gini / Gain Importance")

        plt.tight_layout()
        out_path = self.figure_dir / "feature_importance.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved feature importance plot to {out_path}")
        return out_path
