from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


st.set_page_config(page_title="Mangrove Classifier", page_icon="🌿", layout="wide")

APP_DIR = Path(__file__).resolve().parent
MODEL_OPTIONS = {
    "Setelah Fine Tuning": {
        "path": APP_DIR / "models" / "best_model_finetuned.keras",
        "architecture": "DenseNet121",
        "scenario": "S70 LS0.0",
        "fine_tuning": "Ya",
        "accuracy": "98,61%",
    },
    "Tanpa Fine Tuning": {
        "path": APP_DIR / "models" / "best_model_no_finetuning.keras",
        "architecture": "DenseNet121",
        "scenario": "S70 LS0.0",
        "fine_tuning": "Tidak",
        "accuracy": "98,15%",
    },
}
CLASS_NAMES = ["01 Subur", "02 Terdegradasi", "03 Non-Vegetasi"]
IMG_SIZE = (224, 224)

PALETTE = {
    "bg": "#F6FBF7",
    "primary": "#6BBF8A",
    "soft": "#DFF3E6",
    "dark": "#245C3A",
    "border": "#D8EBDD",
    "text": "#1F2933",
    "muted": "#6B7280",
    "card": "#FFFFFF",
}

CLASS_INFO = {
    "01 Subur": "Area mangrove dengan tutupan vegetasi rapat dan kondisi kanopi baik.",
    "02 Terdegradasi": "Area mangrove dengan vegetasi mulai menurun atau tidak merata.",
    "03 Non-Vegetasi": "Area non-vegetatif seperti tanah terbuka, pasir, lumpur, atau air.",
}


def css():
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

            .stApp {{ background: {PALETTE["bg"]}; color: {PALETTE["text"]}; font-family: 'Inter', sans-serif; }}
            .block-container {{ padding-top: 3.0rem !important; padding-bottom: 1.8rem; }}

            /* ── Hero ── */
            .hero {{
                background: linear-gradient(135deg, #ffffff 0%, {PALETTE["soft"]} 100%);
                border: 1px solid {PALETTE["border"]};
                border-radius: 20px;
                padding: 18px 20px;
                margin-top: 0.35rem;
                margin-bottom: 16px;
            }}
            .title {{
                font-size: 2.1rem;
                font-weight: 800;
                color: {PALETTE["dark"]};
                line-height: 1.15;
            }}
            .subtitle {{
                font-size: 0.98rem;
                color: {PALETTE["muted"]};
                margin-top: 6px;
            }}

            /* ── Card ── */
            .card {{
                background: {PALETTE["card"]};
                border: 1px solid {PALETTE["border"]};
                border-radius: 18px;
                padding: 18px;
                box-shadow: 0 8px 20px rgba(31, 41, 51, 0.05);
                margin-bottom: 14px;
            }}
            .section {{
                font-size: 1rem;
                font-weight: 700;
                color: {PALETTE["dark"]};
                margin-bottom: 10px;
            }}
            .mini {{
                background: #F9FCFA;
                border: 1px solid {PALETTE["border"]};
                border-radius: 14px;
                padding: 10px 12px;
                margin-bottom: 8px;
            }}

            /* ── Model Switch Card ── */
            .switch-card {{
                background: linear-gradient(135deg, #ffffff 0%, {PALETTE["soft"]} 100%);
                border: 1px solid {PALETTE["border"]};
                border-radius: 18px;
                padding: 16px;
                box-shadow: 0 8px 20px rgba(31, 41, 51, 0.05);
                margin-bottom: 12px;
            }}
            .switch-header {{
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 12px;
            }}
            .switch-icon {{
                width: 28px;
                height: 28px;
                background: {PALETTE["primary"]};
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
            }}
            .switch-title {{
                font-size: 1.05rem;
                font-weight: 900;
                color: {PALETTE["dark"]};
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }}
            .switch-subtitle {{
                font-size: 0.92rem;
                font-weight: 600;
                color: {PALETTE["muted"]};
                margin-top: 6px;
                line-height: 1.3;
            }}


            /* ── Pill Toggle ── */
            .pill-track {{
                background: #EAF4EE;
                border: 1px solid {PALETTE["border"]};
                border-radius: 50px;
                padding: 4px;
                display: flex;
                gap: 4px;
                position: relative;
            }}
            .pill-btn {{
                flex: 1;
                padding: 8px 10px;
                border-radius: 46px;
                border: none;
                cursor: pointer;
                font-family: 'Inter', sans-serif;
                font-size: 0.82rem;
                font-weight: 600;
                transition: all 0.22s ease;
                text-align: center;
                outline: none;
            }}
            .pill-btn-active {{
                background: linear-gradient(135deg, {PALETTE["dark"]} 0%, #3a7d56 100%);
                color: #ffffff;
                box-shadow: 0 4px 12px rgba(36, 92, 58, 0.28);
                letter-spacing: 0.01em;
            }}
            .pill-btn-inactive {{
                background: transparent;
                color: {PALETTE["muted"]};
            }}
            .pill-btn-inactive:hover {{
                background: rgba(107, 191, 138, 0.15);
                color: {PALETTE["dark"]};
            }}

            /* ── Active model badge ── */
            .active-badge {{
                display: inline-flex;
                align-items: center;
                gap: 5px;
                margin-top: 10px;
                background: {PALETTE["soft"]};
                border: 1px solid {PALETTE["border"]};
                border-radius: 50px;
                padding: 4px 10px;
                font-size: 0.78rem;
                font-weight: 600;
                color: {PALETTE["dark"]};
            }}
            .dot-active {{
                width: 7px;
                height: 7px;
                background: {PALETTE["primary"]};
                border-radius: 50%;
                display: inline-block;
            }}

            /* ── Misc ── */
            .label {{ font-size: 0.82rem; color: {PALETTE["muted"]}; }}
            .value {{ font-size: 0.98rem; font-weight: 700; color: {PALETTE["dark"]}; }}
            .result {{
                font-size: 1.4rem;
                font-weight: 800;
                color: {PALETTE["dark"]};
            }}
            .hint {{
                background: {PALETTE["soft"]};
                border: 1px solid {PALETTE["border"]};
                border-radius: 14px;
                padding: 12px;
                color: {PALETTE["dark"]};
            }}
            .footer {{
                margin-top: 18px;
                padding-top: 12px;
                border-top: 1px solid {PALETTE["border"]};
                text-align: center;
                color: {PALETTE["muted"]};
                font-size: 0.88rem;
            }}

            /* Hide Streamlit button default style, restyle as pill */
            div[data-testid="column"] .stButton > button {{
                width: 100%;
                border-radius: 46px;
                border: none;
                padding: 9px 10px;
                font-family: 'Inter', sans-serif;
                font-size: 0.82rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s ease;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_best_model(model_path_str):
    model_path = Path(model_path_str)
    if not model_path.exists():
        return None
    return load_model(model_path)


def preprocess_image(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        return None, "File yang diunggah bukan gambar valid."

    preview = image.copy()
    image = image.resize(IMG_SIZE)
    array = img_to_array(image)
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)
    return (preview, array), None


def predict(model, image_array):
    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100
    return predicted_class, confidence, probabilities


def prob_chart(probabilities):
    df = pd.DataFrame(
        {"Kelas": CLASS_NAMES, "Probabilitas (%)": np.round(probabilities * 100, 2)}
    )
    fig = go.Figure(
            go.Bar(
                x=df["Probabilitas (%)"],
                y=df["Kelas"],
                orientation="h",
                marker=dict(color=["#67D68B", "#8EE0A6", "#B9F0C9"]),
                text=df["Probabilitas (%)"].map(lambda x: f"{x:.2f}%"),
                textposition="outside",
                textfont=dict(color="#245C3A", size=13),
            )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 100], gridcolor="#E8F2EA", tickfont=dict(color="#245C3A")),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#245C3A")),
        showlegend=False,
    )
    return fig, df


def render_model_switch():
    """Render model selector as dropdown, but keep existing model naming."""
    model_keys = list(MODEL_OPTIONS.keys())

    # Default selection
    if "selected_model_name" not in st.session_state:
        st.session_state.selected_model_name = model_keys[0]

    st.markdown(
        """
        <div class="switch-card">
            <div class="switch-header">
                <div class="switch-icon">🔀</div>
                <div class="switch-title">Switch Model</div>
            </div>
            <div class="switch-subtitle">Pilih model untuk melakukan prediksi</div>
        """,

        unsafe_allow_html=True,
    )

    selected_name = st.selectbox(
        "Pilih model",
        options=model_keys,
        index=model_keys.index(st.session_state.selected_model_name)
        if st.session_state.selected_model_name in model_keys
        else 0,
    )

    # Persist + rerun if changed
    if selected_name != st.session_state.selected_model_name:
        st.session_state.selected_model_name = selected_name
        st.rerun()

    st.markdown(
        f"""
        <div class="active-badge">
            <span class="dot-active"></span>
            Model Aktif: <b>{selected_name}</b>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return selected_name



def main():
    css()

    st.markdown(
        """
        <div class="hero">
            <div class="title">Klasifikasi Kondisi Tutupan Mangrove</div>
            <div class="subtitle">Visualisasi prediksi DenseNet121 berbasis citra drone untuk skenario S70 LS0.0.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, middle_col, right_col = st.columns([1.0, 1.4, 1.0], gap="medium")

    with left_col:
        # ── Custom pill model switch ──
        selected_model_name = render_model_switch()

        selected_model = MODEL_OPTIONS[selected_model_name]
        model_path = selected_model["path"]
        model = load_best_model(str(model_path))
        if model is None:
            st.error(f"Model tidak ditemukan: {model_path}")
            return

        st.markdown(
            f"""
            <div class="card">
                <div class="section">Ringkasan Model</div>
                <div class="mini"><div class="label">Arsitektur</div><div class="value">{selected_model["architecture"]}</div></div>
                <div class="mini"><div class="label">Skenario</div><div class="value">{selected_model["scenario"]}</div></div>
                <div class="mini"><div class="label">Fine Tuning</div><div class="value">{selected_model["fine_tuning"]}</div></div>
                <div class="mini"><div class="label">Akurasi Uji</div><div class="value">{selected_model["accuracy"]}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with middle_col:
        st.markdown(
            """
            <div class="card">
                <div class="section">Upload Citra</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "JPG, JPEG, PNG",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file is None:
            st.info("Unggah citra mangrove untuk melihat preview.")
        else:
            processed, error = preprocess_image(uploaded_file)
            if error:
                st.warning(error)
            else:
                preview_image, image_array = processed
                st.image(preview_image, width=340)

    with right_col:
        if uploaded_file is None:
            st.markdown(
                """
                <div class="card">
                    <div class="section">Hasil Prediksi</div>
                    <div class="hint">Hasil prediksi akan tampil setelah gambar diunggah.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="card">
                    <div class="section">Keterangan</div>
                    <div class="hint">Silakan unggah citra drone mangrove untuk memulai analisis.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            processed, error = preprocess_image(uploaded_file)
            if error:
                st.warning(error)
            else:
                preview_image, image_array = processed
                try:
                    predicted_class, confidence, probabilities = predict(model, image_array)
                except Exception as exc:
                    st.error(f"Prediksi gagal: {exc}")
                    return

                chart, df = prob_chart(probabilities)

                st.markdown(
                    f"""
                    <div class="card">
                        <div class="section">Hasil Prediksi</div>
                        <div class="result">{predicted_class}</div>
                        <div style="margin-top:6px;color:{PALETTE["muted"]};">Confidence tertinggi: <b>{confidence:.2f}%</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div class="hint">{CLASS_INFO[predicted_class]}</div>', unsafe_allow_html=True)

                st.markdown('<div class="card" style="margin-top:12px;"><div class="section">Probabilitas</div></div>', unsafe_allow_html=True)
                st.plotly_chart(chart, use_container_width=True)
                st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="footer">Aplikasi ini digunakan untuk visualisasi hasil model penelitian klasifikasi kondisi tutupan mangrove berbasis citra drone.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()