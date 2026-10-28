"""
test_preprocessor.py - Unit Tests for Meteorological Preprocessor
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

from src.data.preprocessor import MeteorologicalPreprocessor


@pytest.fixture
def sample_config():
    return {
        "meteorology": {
            "magnus_tetens": {"a": 17.27, "b": 237.7}
        },
        "features": {
            "numeric": [
                "temp_c", "humidity_pct", "pressure_hpa", "wind_speed_ms", "wind_deg",
                "solar_radiation_pct", "dew_point_c", "dew_spread_c",
                "pressure_tendency_3h", "hour_sin", "hour_cos", "month_sin", "month_cos"
            ],
            "target": "rainfall_binary"
        },
        "split": {
            "test_size": 0.20,
            "random_state": 42,
            "stratify": True
        }
    }


def test_magnus_tetens_dew_point_calculation(sample_config):
    preprocessor = MeteorologicalPreprocessor(sample_config)
    # At 100% relative humidity, dew point must equal dry bulb temperature
    temp = np.array([25.0, 30.0])
    rh = np.array([100.0, 100.0])
    dew_points = preprocessor.compute_magnus_tetens_dew_point(temp, rh)
    np.testing.assert_allclose(dew_points, temp, rtol=1e-3)


def test_dew_spread_positive(sample_config):
    preprocessor = MeteorologicalPreprocessor(sample_config)
    temp = np.array([32.0])
    rh = np.array([60.0])
    dew_point = preprocessor.compute_magnus_tetens_dew_point(temp, rh)
    dew_spread = temp - dew_point
    assert dew_spread[0] > 0.0, "Dew spread should be strictly positive for sub-saturated air"


def test_transform_features(sample_config):
    preprocessor = MeteorologicalPreprocessor(sample_config)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01 12:00:00", periods=5, freq="h"),
        "station_id": [96745] * 5,
        "temp_c": [28.0, 29.0, 30.0, 27.5, 26.0],
        "humidity_pct": [75.0, 70.0, 68.0, 85.0, 90.0],
        "pressure_hpa": [1010.0, 1009.5, 1009.0, 1008.2, 1008.0],
        "wind_speed_ms": [3.0, 3.5, 4.0, 2.5, 2.0],
        "wind_deg": [180.0, 190.0, 185.0, 200.0, 210.0],
        "solar_radiation_pct": [60.0, 70.0, 65.0, 20.0, 5.0],
        "rainfall_binary": [0, 0, 0, 1, 1]
    })

    df_trans = preprocessor.transform_features(df)
    assert "dew_point_c" in df_trans.columns
    assert "dew_spread_c" in df_trans.columns
    assert "pressure_tendency_3h" in df_trans.columns
    assert "hour_sin" in df_trans.columns
    assert "hour_cos" in df_trans.columns
