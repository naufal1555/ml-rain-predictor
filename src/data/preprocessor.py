"""
preprocessor.py - Feature Engineering & Atmospheric Thermodynamic Transformations
Author: Ahmad Fathin Naufal (Universitas Gadjah Mada)
Role: AI Engineer

Key Responsibilities:
1. Magnus-Tetens Dew Point calculation.
2. Thermodynamic Dew Point Depression (spread).
3. Barometric pressure tendency (3-hour Delta P).
4. Cyclical temporal encoding (sin/cos of hour & month).
5. Clean scaling and stratified train-test partitioning.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logger = logging.getLogger("preprocessor")


class MeteorologicalPreprocessor:
    """Industrial preprocessor implementing thermodynamic atmospheric feature engineering."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = StandardScaler()
        self.is_fitted = False

        self.a = float(config.get("meteorology", {}).get("magnus_tetens", {}).get("a", 17.27))
        self.b = float(config.get("meteorology", {}).get("magnus_tetens", {}).get("b", 237.7))
        self.feature_cols = config.get("features", {}).get("numeric", [])
        self.target_col = config.get("features", {}).get("target", "rainfall_binary")

    def compute_magnus_tetens_dew_point(self, temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
        """
        Computes dew point temperature (Td) using the Magnus-Tetens formula:
        alpha(T, RH) = (a * T) / (b + T) + ln(RH / 100)
        Td = (b * alpha) / (a - alpha)
        """
        rh_clipped = np.clip(rh_pct, 1.0, 100.0)
        alpha = (self.a * temp_c) / (self.b + temp_c) + np.log(rh_clipped / 100.0)
        dew_point = (self.b * alpha) / (self.a - alpha)
        return dew_point

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies feature transformations to raw meteorological dataframe."""
        df_out = df.copy()

        # Ensure datetime
        if not np.issubdtype(df_out["timestamp"].dtype, np.datetime64):
            df_out["timestamp"] = pd.to_datetime(df_out["timestamp"])

        # 1. Thermodynamic Features
        df_out["dew_point_c"] = self.compute_magnus_tetens_dew_point(
            df_out["temp_c"].values, df_out["humidity_pct"].values
        )
        df_out["dew_spread_c"] = df_out["temp_c"] - df_out["dew_point_c"]

        # 2. Barometric 3-Hour Tendency per station
        df_out = df_out.sort_values(by=["station_id", "timestamp"])
        df_out["pressure_tendency_3h"] = df_out.groupby("station_id")["pressure_hpa"].diff(periods=3)
        df_out["pressure_tendency_3h"] = df_out["pressure_tendency_3h"].fillna(0.0)

        # 3. Cyclical Temporal Encodings (sin/cos preserving continuous circular nature of time)
        hours = df_out["timestamp"].dt.hour.values
        months = df_out["timestamp"].dt.month.values

        df_out["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        df_out["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
        df_out["month_sin"] = np.sin(2 * np.pi * months / 12.0)
        df_out["month_cos"] = np.cos(2 * np.pi * months / 12.0)

        # Impute any residual NaNs with column median
        for col in self.feature_cols:
            if col in df_out.columns and df_out[col].isnull().any():
                df_out[col] = df_out[col].fillna(df_out[col].median())

        return df_out

    def fit_transform_split(
        self, df_raw: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Runs full transformation, fits scaler, and executes stratified train-test split."""
        df_engineered = self.transform_features(df_raw)

        X = df_engineered[self.feature_cols]
        y = df_engineered[self.target_col]

        split_cfg = self.config.get("split", {})
        test_size = float(split_cfg.get("test_size", 0.20))
        random_state = int(split_cfg.get("random_state", 42))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if split_cfg.get("stratify", True) else None
        )

        # Fit Scaler on Training data only (preventing data leakage)
        self.scaler.fit(X_train)
        self.is_fitted = True

        logger.info(f"Feature matrix split: Train {X_train.shape}, Test {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def scale_features(self, X: pd.DataFrame) -> np.ndarray:
        """Applies fitted standard scaler to feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor scaler is not fitted yet.")
        return self.scaler.transform(X[self.feature_cols])

    def save_scaler(self, output_path: Path) -> None:
        """Serializes fitted scaler artifact to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, output_path)
        logger.info(f"StandardScaler persisted to {output_path}")

    def load_scaler(self, input_path: Path) -> None:
        """Loads fitted scaler artifact from disk."""
        self.scaler = joblib.load(input_path)
        self.is_fitted = True
        logger.info(f"StandardScaler loaded from {input_path}")
