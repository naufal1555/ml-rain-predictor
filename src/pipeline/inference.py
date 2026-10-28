"""
inference.py - Production Inference Pipeline for Real-Time BMKG Rainfall Classification
Author: Ahmad Fathin Naufal (Universitas Gadjah Mada)
Role: AI Engineer

Provides real-time single-sample and batch prediction interfaces with automated
feature engineering, Magnus-Tetens dew point calculation, and categorical risk tiering.
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import logging

logger = logging.getLogger("inference_pipeline")


class RainInferencePipeline:
    """End-to-end production inference pipeline for meteorological rainfall prediction."""

    def __init__(self, model_path: Path, scaler_path: Path):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler artifact not found at {self.scaler_path}")

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        # Magnus-Tetens coefficients
        self.a = 17.27
        self.b = 237.7

        self.feature_cols = [
            "temp_c", "humidity_pct", "pressure_hpa", "wind_speed_ms", "wind_deg",
            "solar_radiation_pct", "dew_point_c", "dew_spread_c",
            "pressure_tendency_3h", "hour_sin", "hour_cos", "month_sin", "month_cos"
        ]

    def compute_dew_point(self, temp_c: float, humidity_pct: float) -> float:
        """Magnus-Tetens Dew Point approximation in Celsius."""
        rh_clipped = max(1.0, min(100.0, humidity_pct))
        alpha = (self.a * temp_c) / (self.b + temp_c) + np.log(rh_clipped / 100.0)
        return (self.b * alpha) / (self.a - alpha)

    def predict_single(
        self,
        temp_c: float,
        humidity_pct: float,
        pressure_hpa: float,
        wind_speed_ms: float,
        wind_deg: float,
        solar_radiation_pct: float,
        pressure_tendency_3h: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Executes single-observation real-time prediction.
        Returns prediction label, calibrated probability, and risk tier.
        """
        if timestamp is None:
            timestamp = datetime.now()

        hour = timestamp.hour
        month = timestamp.month

        dew_point_c = self.compute_dew_point(temp_c, humidity_pct)
        dew_spread_c = temp_c - dew_point_c

        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        month_sin = np.sin(2 * np.pi * month / 12.0)
        month_cos = np.cos(2 * np.pi * month / 12.0)

        feature_dict = {
            "temp_c": temp_c,
            "humidity_pct": humidity_pct,
            "pressure_hpa": pressure_hpa,
            "wind_speed_ms": wind_speed_ms,
            "wind_deg": wind_deg,
            "solar_radiation_pct": solar_radiation_pct,
            "dew_point_c": dew_point_c,
            "dew_spread_c": dew_spread_c,
            "pressure_tendency_3h": pressure_tendency_3h,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "month_sin": month_sin,
            "month_cos": month_cos
        }

        df_in = pd.DataFrame([feature_dict])[self.feature_cols]
        X_scaled = self.scaler.transform(df_in)

        pred_class = int(self.model.predict(X_scaled)[0])
        prob_rain = float(self.model.predict_proba(X_scaled)[0, 1])

        # Risk Classification
        if prob_rain < 0.20:
            risk_tier = "Minimal (Clear Weather)"
            alert_color = "#2ECC71"
        elif prob_rain < 0.50:
            risk_tier = "Low (Scattered Clouds)"
            alert_color = "#3498DB"
        elif prob_rain < 0.75:
            risk_tier = "Moderate (High Probability of Rain)"
            alert_color = "#F39C12"
        else:
            risk_tier = "High / Severe Alert (Heavy Precipitation Likely)"
            alert_color = "#E74C3C"

        return {
            "rainfall_prediction": "Rain" if pred_class == 1 else "No Rain",
            "rain_binary": pred_class,
            "rain_probability_pct": round(prob_rain * 100, 2),
            "risk_tier": risk_tier,
            "alert_color": alert_color,
            "dew_point_c": round(dew_point_c, 2),
            "dew_spread_c": round(dew_spread_c, 2),
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }

    def predict_batch(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Executes high-throughput batch prediction over meteorological dataframe."""
        df = df_raw.copy()
        if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Compute thermodynamic features
        df["dew_point_c"] = [
            self.compute_dew_point(t, rh) for t, rh in zip(df["temp_c"], df["humidity_pct"])
        ]
        df["dew_spread_c"] = df["temp_c"] - df["dew_point_c"]

        if "pressure_tendency_3h" not in df.columns:
            df = df.sort_values(by=["station_id", "timestamp"])
            df["pressure_tendency_3h"] = df.groupby("station_id")["pressure_hpa"].diff(periods=3).fillna(0.0)

        hours = df["timestamp"].dt.hour.values
        months = df["timestamp"].dt.month.values
        df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
        df["month_sin"] = np.sin(2 * np.pi * months / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * months / 12.0)

        X = df[self.feature_cols]
        X_scaled = self.scaler.transform(X)

        df["pred_rainfall_binary"] = self.model.predict(X_scaled)
        df["pred_rain_prob_pct"] = np.round(self.model.predict_proba(X_scaled)[:, 1] * 100, 2)
        df["pred_label"] = np.where(df["pred_rainfall_binary"] == 1, "Rain", "No Rain")

        return df
