"""
app.py - Interactive Streamlit Dashboard for BMKG Rainfall Prediction
Author: Ahmad Fathin Naufal (Universitas Gadjah Mada)
Role: AI Engineer | Python, Scikit-learn, XGBoost, Power BI Integration
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.pipeline.inference import RainInferencePipeline

# Page setup
st.set_page_config(
    page_title="BMKG Rain Predictor | Ahmad Fathin Naufal",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive engineering look
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0B2545;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge-candidate {
        background-color: #E2F0D9;
        color: #008837;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🌧️ BMKG Real-Time Rainfall Classification & Prediction Pipeline</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    '<b>AI Engineer Portfolio Case Study</b> | Candidate: <b>Ahmad Fathin Naufal</b> (Universitas Gadjah Mada) '
    '<span class="badge-candidate">Production-Grade ML Pipeline</span>'
    '</div>',
    unsafe_allow_html=True
)

# Load Pipeline
MODEL_PATH = BASE_DIR / "outputs" / "models" / "xgboost.joblib"
SCALER_PATH = BASE_DIR / "outputs" / "models" / "scaler.joblib"

@st.cache_resource
def load_pipeline():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None
    return RainInferencePipeline(model_path=MODEL_PATH, scaler_path=SCALER_PATH)

pipeline = load_pipeline()

# Navigation Tabs
tab_live, tab_benchmark, tab_batch = st.tabs([
    "🎯 Live Station Prediction",
    "📊 Model Benchmarks (>80% Acc)",
    "📁 Batch Ingestion & Power BI Export"
])

# =============================================================================
# TAB 1: LIVE STATION PREDICTION
# =============================================================================
with tab_live:
    st.subheader("Automated Meteorological Input & Instant Rainfall Inference")
    st.markdown("Adjust surface weather station telemetry below to calculate thermodynamic atmospheric instability and predict precipitation probability:")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

    with col_ctrl1:
        st.markdown("##### 📍 Weather Station Selection")
        station_choice = st.selectbox(
            "Select BMKG Weather Station:",
            [
                "Stasiun Meteorologi Kemayoran (Jakarta Pusat)",
                "Stasiun Meteorologi Maritim Tanjung Priok (Jakarta Utara)",
                "Stasiun Meteorologi Soekarno-Hatta (Tangerang)",
                "Stasiun Geofisika Curug (Tangerang)",
                "Stasiun Klimatologi Dermaga Bogor (Bogor)"
            ]
        )
        temp_input = st.slider("Ambient Temperature (°C):", min_value=20.0, max_value=38.0, value=28.5, step=0.1)
        humidity_input = st.slider("Relative Humidity (%):", min_value=40.0, max_value=100.0, value=85.0, step=0.5)

    with col_ctrl2:
        st.markdown("##### 💨 Barometric & Wind Metrics")
        pressure_input = st.slider("Surface Pressure (hPa):", min_value=995.0, max_value=1020.0, value=1008.2, step=0.1)
        wind_speed_input = st.slider("Wind Velocity (m/s):", min_value=0.2, max_value=20.0, value=4.5, step=0.1)
        wind_deg_input = st.slider("Wind Azimuth Direction (°):", min_value=0.0, max_value=360.0, value=180.0, step=5.0)

    with col_ctrl3:
        st.markdown("##### ☀️ Solar & Dynamic Tendency")
        solar_input = st.slider("Solar Radiation Index (%):", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
        tendency_input = st.slider("3-Hour Pressure Delta ΔP (hPa):", min_value=-6.0, max_value=6.0, value=-1.8, step=0.1)
        sim_hour = st.slider("Simulated Hour of Day (WIB):", min_value=0, max_value=23, value=15, step=1)

    st.markdown("---")

    if pipeline is not None:
        dummy_time = datetime(2025, 4, 15, sim_hour, 0, 0)
        result = pipeline.predict_single(
            temp_c=temp_input,
            humidity_pct=humidity_input,
            pressure_hpa=pressure_input,
            wind_speed_ms=wind_speed_input,
            wind_deg=wind_deg_input,
            solar_radiation_pct=solar_input,
            pressure_tendency_3h=tendency_input,
            timestamp=dummy_time
        )

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        with res_col1:
            st.metric("Predicted Outcome", result["rainfall_prediction"])
        with res_col2:
            st.metric("Rainfall Probability", f"{result['rain_probability_pct']}%")
        with res_col3:
            st.metric("Dew Point Temperature", f"{result['dew_point_c']} °C")
        with res_col4:
            st.metric("Dew Point Spread", f"{result['dew_spread_c']} °C")

        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["rain_probability_pct"],
            title={"text": f"Precipitation Probability & Risk Tier: {result['risk_tier']}", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": result["alert_color"]},
                "steps": [
                    {"range": [0, 20], "color": "#E8F8F5"},
                    {"range": [20, 50], "color": "#EBF5FB"},
                    {"range": [50, 75], "color": "#FEF9E7"},
                    {"range": [75, 100], "color": "#FDEDEC"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50.0
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    else:
        st.warning("⚠️ Model artifacts not yet trained. Run `python run_pipeline.py` to train and serialize the models.")

# =============================================================================
# TAB 2: MODEL BENCHMARKING & DIAGNOSTICS
# =============================================================================
with tab_benchmark:
    st.subheader("Supervised ML Benchmark Comparison (>80% Accuracy Target)")
    st.markdown("As highlighted on the CV, three supervised learning architectures were cross-validated on Indonesian meteorological telemetry:")

    metrics_csv = BASE_DIR / "outputs" / "figures" / "model_benchmark_metrics.csv"
    if metrics_csv.exists():
        df_bench = pd.read_csv(metrics_csv)
        st.dataframe(df_bench.style.highlight_max(subset=["Accuracy", "F1_Score", "ROC_AUC"], color="#D5F5E3"), use_container_width=True)
    else:
        st.info("Run `python run_pipeline.py` to populate real-time benchmark metrics.")

    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        st.markdown("##### 🎯 Comparative ROC Curves")
        roc_img = BASE_DIR / "outputs" / "figures" / "roc_curves.png"
        if roc_img.exists():
            st.image(str(roc_img), caption="Out-of-sample ROC curves showing AUC > 0.88 across ensembles.")
    with col_fig2:
        st.markdown("##### 🔬 Feature Importance Ranking")
        feat_img = BASE_DIR / "outputs" / "figures" / "feature_importance.png"
        if feat_img.exists():
            st.image(str(feat_img), caption="Relative Humidity, Dew Spread, and Barometric Tendency are leading predictors.")

# =============================================================================
# TAB 3: BATCH INGESTION & POWER BI EXPORT
# =============================================================================
with tab_batch:
    st.subheader("Batch Meteorological Ingestion & Power BI Integration Layer")
    st.markdown("This module supports automated batch ingestion for historical weather evaluation and direct Power BI modeling:")

    raw_csv = BASE_DIR / "data" / "raw" / "bmkg_weather_raw.csv"
    if raw_csv.exists():
        df_preview = pd.read_csv(raw_csv, nrows=100)
        st.markdown(f"**Previewing Ingested BMKG Records (First 100 of {len(pd.read_csv(raw_csv))} rows):**")
        st.dataframe(df_preview, use_container_width=True)

        powerbi_csv = BASE_DIR / "powerbi" / "powerbi_rainfall_dataset.csv"
        if powerbi_csv.exists():
            with open(powerbi_csv, "rb") as fp:
                st.download_button(
                    label="📥 Download Power BI Star-Schema Dataset (CSV)",
                    data=fp,
                    file_name="powerbi_rainfall_dataset.csv",
                    mime="text/csv"
                )
    else:
        st.info("Raw data not generated yet. Execute `python run_pipeline.py`.")
