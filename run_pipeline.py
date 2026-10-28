"""
run_pipeline.py - Master Automated End-to-End ML Pipeline Orchestrator
Author: Ahmad Fathin Naufal (Universitas Gadjah Mada)
Role: AI Engineer | Python, Scikit-learn, XGBoost, Power BI Integration

Executes complete pipeline:
1. Data Ingestion & Quality Validation (BMKG Stations)
2. Atmospheric Thermodynamic Feature Engineering (Magnus-Tetens)
3. 5-Fold Stratified Cross-Validation & Model Training (DT, RF, XGBoost)
4. Out-of-Sample Statistical Evaluation (>80% Accuracy Target)
5. Diagnostic Plot Rendering (Confusion Matrices, ROC, Feature Importance)
6. Star-Schema Dataset Export for Microsoft Power BI
"""

from pathlib import Path
import yaml
import logging
import pandas as pd
import numpy as np
import joblib

from src.data.bmkg_loader import BMKGDataLoader
from src.data.preprocessor import MeteorologicalPreprocessor
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.pipeline.inference import RainInferencePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("orchestrator")


def main():
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "config" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("=" * 75)
    logger.info("STARTING BMKG MACHINE LEARNING RAINFALL PIPELINE")
    logger.info("Author: Ahmad Fathin Naufal | Candidate: AI Engineer (UGM)")
    logger.info("=" * 75)

    # 1. Ingestion
    raw_path = base_dir / config["paths"]["raw_data"]
    loader = BMKGDataLoader(raw_path)
    df_raw = loader.load_or_fetch(force_refresh=True)

    # 2. Preprocessing & Feature Engineering
    preprocessor = MeteorologicalPreprocessor(config)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform_split(df_raw)

    # Persist Scaler
    scaler_path = base_dir / config["paths"]["model_dir"] / "scaler.joblib"
    preprocessor.save_scaler(scaler_path)

    # Scale feature matrices
    X_train_scaled = preprocessor.scale_features(X_train)
    X_test_scaled = preprocessor.scale_features(X_test)

    # 3. Model Training & 5-Fold Stratified Cross-Validation
    trainer = ModelTrainer(config)
    trainer.model_dir = base_dir / config["paths"]["model_dir"]
    cv_results = trainer.cross_validate_all(X_train_scaled, y_train.values, n_splits=5)
    trained_models = trainer.train_and_persist(X_train_scaled, y_train.values)

    # 4. Out-of-Sample Test Set Evaluation
    figure_dir = base_dir / config["paths"]["figure_dir"]
    evaluator = ModelEvaluator(figure_dir=figure_dir, feature_names=preprocessor.feature_cols)

    df_metrics = evaluator.evaluate_test_set(trained_models, X_test_scaled, y_test.values)
    metrics_csv_path = figure_dir / "model_benchmark_metrics.csv"
    df_metrics.to_csv(metrics_csv_path, index=False)
    logger.info(f"Model benchmark metrics saved to {metrics_csv_path}")

    # 5. Diagnostic Figures
    evaluator.plot_confusion_matrices(trained_models, X_test_scaled, y_test.values)
    evaluator.plot_roc_curves(trained_models, X_test_scaled, y_test.values)
    evaluator.plot_feature_importance(trained_models)

    # 6. Verify Accuracy Target (>80% as claimed in CV)
    best_acc = df_metrics["Accuracy"].max()
    best_model_name = df_metrics.iloc[0]["Model"]
    logger.info("=" * 75)
    logger.info(f"BEST MODEL: {best_model_name} with Accuracy = {best_acc * 100:.2f}%")
    if best_acc >= 0.80:
        logger.info("SUCCESS: Model accuracy strictly satisfies the CV claim (>80% accuracy)!")
    else:
        logger.warning(f"Accuracy {best_acc * 100:.2f}% is below 80% threshold.")
    logger.info("=" * 75)

    # 7. Power BI Dataset Export
    powerbi_dir = base_dir / config["paths"]["powerbi_dir"]
    powerbi_dir.mkdir(parents=True, exist_ok=True)
    powerbi_path = base_dir / config["paths"]["powerbi_dataset"]

    best_model_path = base_dir / config["paths"]["model_dir"] / f"{best_model_name.lower()}.joblib"
    inference_engine = RainInferencePipeline(model_path=best_model_path, scaler_path=scaler_path)

    df_powerbi = inference_engine.predict_batch(df_raw)
    df_powerbi.to_csv(powerbi_path, index=False)
    logger.info(f"Power BI star-schema dataset exported to {powerbi_path} ({len(df_powerbi)} records).")

    logger.info("=" * 75)
    logger.info("ML RAINFALL PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
