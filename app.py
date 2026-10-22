import os
from pathlib import Path
from PIL import Image

import streamlit as st
import pandas as pd
import altair as alt

from model_utils import (
    load_model_from_file,
    predict_polymer,
    CLASSES,
    POLYMER_INFO,
    CustomCNN,
    ConvBlock
)

st.set_page_config(
    page_title="Polymer Resin Classifier",
    page_icon="♼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Streamlined, Minimalist CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.7rem 1.2rem;
        border-radius: 12px;
        background: #111827;
        border: 1px solid #1F2937;
        margin-bottom: 1.5rem;
    }

    .app-title {
        font-size: 1.35rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        color: #F9FAFB;
    }

    .api-badge {
        font-size: 0.78rem;
        background: #064E3B;
        color: #34D399;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        text-decoration: none;
    }

    .result-box {
        padding: 1.5rem;
        border-radius: 14px;
        background: #111827;
        border: 1px solid #1F2937;
    }

    .code-tag {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.6rem;
    }

    .exact-name {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.15;
        margin: 0 0 0.4rem 0;
    }

    .score-line {
        font-size: 1.05rem;
        color: #9CA3AF;
        margin-bottom: 1rem;
    }

    .info-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin-top: 1rem;
    }

    .info-card {
        padding: 0.75rem 0.9rem;
        border-radius: 8px;
        background: #1F2937;
        border: 1px solid #374151;
    }

    .info-card-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        color: #9CA3AF;
        font-weight: 700;
        letter-spacing: 0.04em;
    }

    .info-card-val {
        font-size: 0.88rem;
        font-weight: 500;
        color: #E5E7EB;
        margin-top: 0.15rem;
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.3rem 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Clean Navbar
st.markdown("""
<div class="app-header">
    <div class="app-title">♼ Polymer Resin Classifier</div>
    <a href="http://localhost:8000/docs" target="_blank" class="api-badge">FastAPI Docs</a>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_default_model(model_name: str):
    path = Path("models/vgg16.pkl" if model_name == "VGG-16" else "models/custom_cnn.pkl")
    return load_model_from_file(path)


col_left, col_right = st.columns([1, 1.3], gap="large")

with col_left:
    st.markdown("##### 1. Model")
    model_mode = st.radio(
        "Mode",
        ["Default Models", "Upload Custom Model"],
        horizontal=True,
        label_visibility="collapsed"
    )

    model = None
    classes = CLASSES

    if model_mode == "Default Models":
        choice = st.selectbox(
            "Select Model",
            ["VGG-16", "Custom CNN"],
            label_visibility="collapsed"
        )
        model, classes = load_default_model(choice)
    else:
        custom_file = st.file_uploader(
            "Upload Model",
            type=["pkl", "pt", "pth"],
            label_visibility="collapsed"
        )
        if custom_file:
            try:
                model, classes = load_model_from_file(custom_file, filename=custom_file.name)
                st.success(f"Loaded: {custom_file.name}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("##### 2. Resin Image")
    
    # 1-click test samples
    btn_cols = st.columns(7)
    sample_key = None
    samples = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "OTHER"]
    
    for i, s in enumerate(samples):
        with btn_cols[i]:
            if st.button(f"{POLYMER_INFO[s]['symbol']} {s}", key=f"s_{s}"):
                sample_key = s

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    active_image = None
    if sample_key:
        sample_path = next((Path("dataset/test") / sample_key).glob("*.jpg"), None)
        if sample_path:
            active_image = Image.open(sample_path)
            st.session_state["active_img"] = active_image
            st.session_state["sample_lbl"] = sample_key
    elif uploaded_file:
        active_image = Image.open(uploaded_file)
        st.session_state["active_img"] = active_image
        st.session_state["sample_lbl"] = None
    elif "active_img" in st.session_state:
        active_image = st.session_state["active_img"]

with col_right:
    if active_image is not None and model is not None:
        img_box, _ = st.columns([1, 1])
        with img_box:
            st.image(active_image, use_container_width=True)

        res = predict_polymer(model, active_image, classes)
        color = res["color"]

        st.markdown(f"""
        <div class="result-box">
            <span class="code-tag" style="background: {color}22; color: {color}; border: 1px solid {color}66;">
                {res['symbol']} Resin Code #{res['resin_code']} &bull; {res['short_name']}
            </span>
            <div class="exact-name" style="color: {color};">
                {res['exact_name']}
            </div>
            <div class="score-line">
                Confidence: <strong>{res['confidence']:.1f}%</strong>
            </div>
            <div class="info-row">
                <div class="info-card">
                    <div class="info-card-label">Applications</div>
                    <div class="info-card-val">{res['common_uses']}</div>
                </div>
                <div class="info-card">
                    <div class="info-card-label">Recyclability</div>
                    <div class="info-card-val">{res['recyclability']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probabilities Chart
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        prob_df = pd.DataFrame(res["probabilities"])
        prob_df["Label"] = prob_df.apply(lambda r: f"{r['symbol']} Code {r['resin_code']} ({r['class_name']})", axis=1)

        chart = alt.Chart(prob_df).mark_bar(cornerRadius=4, height=18).encode(
            x=alt.X("probability:Q", title="Probability (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Label:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.class_name == res["predicted_class"],
                alt.value(color),
                alt.value("#374151")
            ),
            tooltip=["exact_name:N", "resin_code:Q", alt.Tooltip("probability:Q", format=".1f", title="Confidence (%)")]
        ).properties(height=210)

        st.altair_chart(chart, use_container_width=True)

    elif active_image is None:
        st.markdown("""
        <div class="result-box" style="text-align: center; padding: 3rem 1rem;">
            <div style="font-size: 2.8rem; margin-bottom: 0.5rem; color: #6B7280;">♼</div>
            <h4 style="margin: 0; color: #9CA3AF;">Upload or select a polymer resin image to predict</h4>
        </div>
        """, unsafe_allow_html=True)
    elif model is None:
        st.warning("Please upload or select a model.")
