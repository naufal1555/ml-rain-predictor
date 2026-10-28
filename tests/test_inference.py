"""
test_inference.py - Unit Tests for Inference Pipeline
"""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime

from src.pipeline.inference import RainInferencePipeline


def test_dew_point_logic():
    # Test dew point computation helper without loading models
    pipeline = RainInferencePipeline.__new__(RainInferencePipeline)
    pipeline.a = 17.27
    pipeline.b = 237.7
    dp = pipeline.compute_dew_point(temp_c=25.0, humidity_pct=80.0)
    assert 20.0 < dp < 25.0, "Dew point at 80% RH should be ~21.3 C"
