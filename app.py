import os
import streamlit as st
import numpy as np
import joblib
import time
import plotly.graph_objects as go

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Sonar Signal Classifier",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS — gradient background, animated header, styled cards
# ------------------------------------------------------------------
st.markdown("""
<style>
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.5); }
    70%  { box-shadow: 0 0 0 18px rgba(56, 189, 248, 0); }
    100% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
}
@keyframes sonarPing {
    0%   { transform: scale(0.3); opacity: 0.9; }
    100% { transform: scale(3); opacity: 0; }
}

.stApp {
    background: linear-gradient(-45deg, #04101d, #072744, #041c30, #0b3350);
    background-size: 400% 400%;
    animation: gradientShift 18s ease infinite;
}

.main-header {
    text-align: center;
    padding: 1.2rem 0 0.4rem 0;
    animation: fadeInUp 0.8s ease-out;
}
.main-header h1 {
    background: linear-gradient(90deg, #38bdf8, #818cf8, #38bdf8);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 4s linear infinite;
    font-size: 2.6rem;
    font-weight: 800;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 1.5rem;
    animation: fadeInUp 1s ease-out;
}

.sonar-wrap {
    position: relative;
    width: 90px; height: 90px;
    margin: 0 auto 1rem auto;
    display: flex; align-items: center; justify-content: center;
}
.sonar-dot {
    width: 16px; height: 16px;
    background: #38bdf8;
    border-radius: 50%;
    animation: pulse 2s infinite;
    z-index: 2;
}
.sonar-ring {
    position: absolute;
    width: 16px; height: 16px;
    border-radius: 50%;
    border: 2px solid #38bdf8;
    animation: sonarPing 2s infinite;
}

.result-card {
    padding: 1.5rem 2rem;
    border-radius: 16px;
    text-align: center;
    animation: fadeInUp 0.6s ease-out;
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 1rem;
}
.rock-card {
    background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(34,197,94,0.05));
    border: 1px solid rgba(34,197,94,0.5);
    color: #4ade80;
}
.mine-card {
    background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(239,68,68,0.05));
    border: 1px solid rgba(239,68,68,0.5);
    color: #f87171;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.8rem;
    animation: fadeInUp 0.7s ease-out;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown("""
<div class="main-header"><h1>⚓ Sonar Signal Classifier</h1></div>
<div class="subtitle">Rock vs. Mine detection from 60-band sonar frequency returns</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Load model (cached so it isn't reloaded on every rerun)
# ------------------------------------------------------------------
# Get the directory where app.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rock_vs_mine_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except FileNotFoundError:
    st.error(f"Model file not found at `{MODEL_PATH}`. Place `rock_vs_mine_model.pkl` next to this script.")
    st.stop()

SAMPLE_ROCK = "0.02,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.066,0.2273,0.31,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.555,0.6711,0.6415,0.7104,0.808,0.6791,0.6393,0.5787,0.4847,0.3441,0.201,0.2107,0.1911,0.125,0.2104,0.1235,0.035,0.076,0.04,0.03,0.015,0.02,0.015,0.01,0.005,0.002,0.001,0.002,0.003,0.004,0.002,0.002,0.001,0.002,0.003,0.001,0.002"
SAMPLE_MINE = "0.0453,0.0523,0.0843,0.0689,0.1183,0.2583,0.2156,0.3481,0.3337,0.2872,0.4918,0.6552,0.6919,0.7797,0.7464,0.9444,1.0,0.8874,0.8024,0.7818,0.5212,0.4052,0.3957,0.3914,0.325,0.32,0.3271,0.2767,0.4423,0.2028,0.3788,0.2947,0.1984,0.2341,0.1306,0.4182,0.3835,0.1057,0.184,0.197,0.1674,0.0583,0.1401,0.1628,0.0621,0.0203,0.053,0.0742,0.0409,0.0061,0.0125,0.0084,0.0089,0.0048,0.0094,0.0191,0.014,0.0049,0.0052,0.0044"

if "input_data" not in st.session_state:
    st.session_state.input_data = ""

# ------------------------------------------------------------------
# Sidebar — controls and sample loaders
# ------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Controls")
    st.write("Load a preset or paste your own 60 comma-separated readings.")

    if st.button("🧱 Load Sample Rock Data", use_container_width=True):
        st.session_state.input_data = SAMPLE_ROCK
    if st.button("💣 Load Sample Mine Data", use_container_width=True):
        st.session_state.input_data = SAMPLE_MINE
    if st.button("🧹 Clear", use_container_width=True):
        st.session_state.input_data = ""

    st.divider()
    st.caption("Each reading is a normalized energy value (0–1) within a particular frequency band, integrated over a certain period of time.")

# ------------------------------------------------------------------
# Main input area
# ------------------------------------------------------------------
col_input, col_viz = st.columns([1, 1.2], gap="large")

with col_input:
    input_data = st.text_area(
        "60 Frequency Readings",
        placeholder="0.02, 0.0371, 0.0428, 0.0207, ...",
        height=160,
        key="input_data",
    )

    classify_clicked = st.button("🔍 Classify Signal", use_container_width=True, type="primary")

# Live preview chart of whatever is currently typed/loaded
def parse_features(raw: str):
    values = [float(x.strip()) for x in raw.split(",") if x.strip() != ""]
    return values

with col_viz:
    st.markdown("**Signal Preview**")
    try:
        preview_vals = parse_features(st.session_state.input_data) if st.session_state.input_data else []
    except ValueError:
        preview_vals = []

    if preview_vals:
        fig_preview = go.Figure()
        fig_preview.add_trace(go.Scatter(
            y=preview_vals,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#38bdf8", width=2),
            fillcolor="rgba(56,189,248,0.15)",
        ))
        fig_preview.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1",
            xaxis=dict(showgrid=False, title="Frequency band"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", range=[0, 1]),
        )
        st.plotly_chart(fig_preview, use_container_width=True)
    else:
        st.info("Waveform preview will appear here once data is entered.")

st.divider()

# ------------------------------------------------------------------
# Classification
# ------------------------------------------------------------------
if classify_clicked:
    raw = st.session_state.input_data
    try:
        values = parse_features(raw)
        features = np.array(values).reshape(1, -1)

        if features.shape[1] != 60:
            st.error(f"Expected 60 feature values, but received {features.shape[1]}.")
        else:
            # Animated "scanning" placeholder before revealing the result
            scan_placeholder = st.empty()
            scan_placeholder.markdown("""
            <div class="sonar-wrap">
                <div class="sonar-ring"></div>
                <div class="sonar-dot"></div>
            </div>
            <p style="text-align:center; color:#94a3b8;">Scanning sonar returns…</p>
            """, unsafe_allow_html=True)
            time.sleep(1.1)
            scan_placeholder.empty()

            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]

            classes = list(model.classes_)
            pred_index = classes.index(prediction)
            confidence = probability[pred_index]

            is_rock = prediction == "R"
            label = "Rock" if is_rock else "Mine"
            emoji = "🧱" if is_rock else "💣"
            card_class = "rock-card" if is_rock else "mine-card"

            res_col1, res_col2 = st.columns([1, 1.2], gap="large")

            with res_col1:
                st.markdown(f"""
                <div class="result-card {card_class}">
                    {emoji} Result: {label}<br>
                    <span style="font-size:1rem; font-weight:400; opacity:0.85;">
                        Confidence: {confidence:.2%}
                    </span>
                </div>
                """, unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Prediction", label)
                m2.metric("Confidence", f"{confidence:.1%}")
                m3.metric("Signal Peak", f"{max(values):.3f}")

                if is_rock:
                    st.balloons()
                else:
                    st.snow()

            with res_col2:
                # Confidence gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    number={"suffix": "%", "font": {"color": "#e2e8f0"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                        "bar": {"color": "#4ade80" if is_rock else "#f87171"},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 50], "color": "rgba(255,255,255,0.05)"},
                            {"range": [50, 100], "color": "rgba(255,255,255,0.1)"},
                        ],
                    },
                    title={"text": "Confidence", "font": {"color": "#94a3b8", "size": 14}},
                ))
                fig_gauge.update_layout(
                    height=220,
                    margin=dict(l=20, r=20, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Full frequency-band bar chart, colored by amplitude
            st.markdown("**Full Frequency Band Breakdown**")
            fig_bars = go.Figure(go.Bar(
                x=list(range(1, 61)),
                y=values,
                marker=dict(
                    color=values,
                    colorscale="Blues" if is_rock else "Reds",
                    line=dict(width=0),
                ),
            ))
            fig_bars.update_layout(
                height=260,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                xaxis=dict(title="Band #", showgrid=False),
                yaxis=dict(title="Energy", showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            )
            st.plotly_chart(fig_bars, use_container_width=True)

    except ValueError as e:
        st.error(f"Invalid input format: {e}")