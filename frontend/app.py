import streamlit as st
import requests
from PIL import Image
import time

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ForestVision AI - Satellite Analytics",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS (Dark Emerald Theme & Rounded Cards)
st.markdown("""
<style>
    /* Main Background Accent */
    .main {
        background-color: #0b0f12;
    }
    
    /* Card Container Styling */
    div[data-testid="stVerticalBlock"] > div[data-testid="stBlock"] {
        border-radius: 12px;
    }
    
    /* Custom Metric Styling */
    .metric-card {
        background-color: #161c22;
        border: 1px solid #28333f;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    
    /* Result Badges */
    .badge-forest {
        background-color: rgba(34, 197, 94, 0.15);
        border: 1px solid #22c55e;
        color: #4ade80;
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
    }
    
    .badge-nonforest {
        background-color: rgba(245, 158, 11, 0.15);
        border: 1px solid #f59e0b;
        color: #fbbf24;
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
    }

    /* Analyze Button Pulse Accent */
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        height: 3.2em !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000/classify"
HEALTH_URL = "http://127.0.0.1:8000/health"

# -----------------------------------------------------------------------------
# 2. SIDEBAR - SYSTEM STATUS & CONTROL PANEL
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌲 ForestVision AI")
    st.caption("Deep Learning Earth Observation Framework")
    st.divider()

    # Backend API Health Check
    st.markdown("#### ⚡ System Health")
    try:
        health_resp = requests.get(HEALTH_URL, timeout=2)
        if health_resp.status_code == 200:
            st.success("API Status: **Online** 🟢")
        else:
            st.warning("API Status: **Degraded** 🟡")
    except Exception:
        st.error("API Status: **Offline** 🔴")

    st.markdown("**Endpoint:** `localhost:8000/classify`")
    st.markdown("**Input Compatibility:** Sentinel-2, Landsat (PNG/JPG)")

    st.divider()
    
    st.markdown("#### 🛰️ Target Classes")
    st.markdown("""
    * 🌲 **Forest:** Dense Canopy, Rainforest, Woodland
    * 🚜 **Non-Forest:** Urban Grid, Agricultural Land, Bare Soil
    """)
    
    st.divider()
    st.caption("ForestVision v1.0 • PyTorch & FastAPI Engine")

# -----------------------------------------------------------------------------
# 3. MAIN DASHBOARD HEADER
# -----------------------------------------------------------------------------
st.title("🛰️ ForestVision - Satellite Image Classifier")
st.markdown("Real-time automated land cover classification using Deep Learning.")
st.divider()

# Tabs Navigation
tab1, tab2 = st.tabs(["🔍 Patch Inference", "ℹ️ Model Architecture"])

# -----------------------------------------------------------------------------
# TAB 1: INFERENCE ENGINE
# -----------------------------------------------------------------------------
with tab1:
    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        with st.container(border=True):
            st.subheader("📤 Upload Patch")
            uploaded_file = st.file_uploader(
                "Choose a satellite image patch...", 
                type=["png", "jpg", "jpeg"],
                help="Recommended patch size: 256x256 or 512x512 pixels"
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded Patch ({image.width}x{image.height} px)", use_container_width=True)
                
                analyze_btn = st.button("🚀 Analyze Satellite Patch", use_container_width=True)

    with col_output:
        with st.container(border=True):
            st.subheader("📊 Inference Results")
            
            if uploaded_file is not None and analyze_btn:
                start_time = time.time()
                
                with st.spinner("Processing deep learning inference pipeline..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        response = requests.post(API_URL, files=files)
                        latency = (time.time() - start_time) * 1000  # Latency in ms

                        if response.status_code == 200:
                            result = response.json()

                            # Extract result keys securely
                            label = str(result.get("label") or result.get("class") or result.get("prediction") or "Unknown")
                            confidence = float(result.get("confidence") or result.get("probability") or result.get("score") or 0.0)

                            # Normalize confidence float
                            if confidence <= 1.0:
                                conf_pct = confidence * 100
                                conf_float = confidence
                            else:
                                conf_pct = confidence
                                conf_float = confidence / 100.0

                            st.write("### ")
                            
                            # Display Classification Badge
                            if "forest" in label.lower() and "non" not in label.lower():
                                st.markdown(
                                    f'<div class="badge-forest">🌲 PREDICTED: {label.upper()}</div>', 
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f'<div class="badge-nonforest">🚜 PREDICTED: {label.upper()}</div>', 
                                    unsafe_allow_html=True
                                )

                            st.write("### ")

                            # Metrics Row
                            m1, m2 = st.columns(2)
                            with m1:
                                st.metric("Model Certainty", f"{conf_pct:.2f}%")
                            with m2:
                                st.metric("Inference Speed", f"{latency:.1f} ms")

                            # Confidence Meter
                            st.markdown("**Confidence Probability:**")
                            st.progress(min(max(conf_float, 0.0), 1.0))

                            # API Raw Payload Drawer
                            st.divider()
                            with st.expander("🛠️ View API JSON Payload"):
                                st.json({
                                    "status_code": 200,
                                    "latency_ms": round(latency, 2),
                                    "model_response": result
                                })

                        else:
                            st.error(f"❌ Server Error ({response.status_code}): {response.text}")

                    except requests.exceptions.ConnectionError:
                        st.error("❌ Connection Failed! Ensure FastAPI backend is running on `http://127.0.0.1:8000`.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
            
            elif uploaded_file is None:
                st.info("👈 Upload a satellite patch on the left side to execute the AI model.")
            else:
                st.info("👆 Click **Analyze Satellite Patch** to execute classification.")

# -----------------------------------------------------------------------------
# TAB 2: MODEL ARCHITECTURE INFO
# -----------------------------------------------------------------------------
with tab2:
    with st.container(border=True):
        st.subheader("🧠 Deep Learning Pipeline Overview")
        st.markdown("""
        **ForestVision** leverages Convolutional Neural Network (CNN) feature extraction to process spatial imagery and predict canopy density.

        ---
        #### Key Technical Specifications:
        * **Input Resolution:** RGB 3-Channel Imagery
        * **Feature Extraction:** Deep Feature Backbone
        * **Output Classes:** Binary (`Forest` vs. `Non-Forest`)
        * **Backend Framework:** FastAPI / Uvicorn Execution Server
        * **Frontend Interface:** Streamlit Engine
        """)