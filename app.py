"""
Frontend con Streamlit para estimar riesgo de odio en texto.
"""
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import DATA_DIR
from data_loader import DataLoader
from predictor import HateSpeechPredictor
from preprocessor import TextPreprocessor


st.set_page_config(
    page_title="Radar de Discurso",
    page_icon="R",
    layout="wide",
)


@st.cache_resource
def build_predictor():
    """Entrena el modelo base una vez por sesi\u00f3n."""
    loader = DataLoader(DATA_DIR)
    df = loader.create_sample_dataset(n_samples=1400)
    preprocessor = TextPreprocessor()
    df = preprocessor.preprocess_dataframe(df, "text")

    predictor = HateSpeechPredictor()
    predictor.fit(df["text"].tolist(), df["label"].tolist())
    return predictor, preprocessor


predictor, preprocessor = build_predictor()

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 14% 18%, rgba(70, 133, 255, 0.16), transparent 26%),
            radial-gradient(circle at 88% 16%, rgba(66, 211, 255, 0.10), transparent 22%),
            linear-gradient(180deg, #08101d 0%, #0d1726 100%);
        color: #edf4ff;
    }
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2.5rem;
        max-width: 1180px;
    }
    .hero {
        padding: 1.9rem 2rem;
        border-radius: 32px;
        background:
            linear-gradient(135deg, rgba(18, 31, 53, 0.92), rgba(14, 24, 40, 0.88));
        border: 1px solid rgba(141, 166, 204, 0.15);
        box-shadow: 0 26px 80px rgba(0, 0, 0, 0.34);
        margin-bottom: 1.2rem;
        backdrop-filter: blur(18px);
    }
    .hero-kicker {
        margin: 0 0 0.4rem;
        font-size: 0.76rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #7fd9ff;
        font-weight: 700;
    }
    .hero-title {
        margin: 0;
        font-size: clamp(2.7rem, 6vw, 4.6rem);
        line-height: 0.92;
        max-width: 12ch;
        color: #edf4ff;
        letter-spacing: -0.04em;
    }
    .hero-copy {
        margin-top: 0.9rem;
        color: #9eb1ca;
        max-width: 68ch;
        font-size: 1.02rem;
        line-height: 1.6;
    }
    .panel {
        padding: 1.2rem 1.2rem 1rem;
        border-radius: 26px;
        background: rgba(16, 26, 43, 0.78);
        border: 1px solid rgba(141, 166, 204, 0.14);
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.24);
        backdrop-filter: blur(18px);
    }
    .result-card {
        padding: 1.2rem 1.2rem 1rem;
        border-radius: 22px;
        border: 1px solid rgba(141, 166, 204, 0.12);
        background:
            linear-gradient(180deg, rgba(20, 34, 56, 0.94), rgba(14, 24, 38, 0.96));
    }
    .scoreboard {
        margin-top: 1rem;
        display: grid;
        gap: 0.9rem;
    }
    .shell-card {
        margin-top: 1rem;
        padding: 1rem 1.05rem;
        border-radius: 22px;
        border: 1px solid rgba(141, 166, 204, 0.12);
        background:
            linear-gradient(180deg, rgba(20, 34, 56, 0.94), rgba(14, 24, 38, 0.96));
    }
    .score-track {
        display: grid;
        gap: 0.45rem;
        padding: 0.85rem 0.95rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid rgba(141, 166, 204, 0.08);
    }
    .score-track header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #dce7f5;
        font-size: 0.95rem;
    }
    .score-track small {
        color: #90a6c3;
        font-size: 0.8rem;
        line-height: 1.45;
    }
    .score-bar {
        position: relative;
        height: 14px;
        border-radius: 999px;
        overflow: hidden;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(141, 166, 204, 0.08);
    }
    .score-fill {
        position: absolute;
        inset: 0 auto 0 0;
        border-radius: 999px;
    }
    .weight-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        width: fit-content;
        padding: 0.32rem 0.6rem;
        border-radius: 999px;
        background: rgba(127, 217, 255, 0.08);
        border: 1px solid rgba(127, 217, 255, 0.16);
        color: #bcecff;
        font-size: 0.75rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .shell-note {
        margin-top: 0.1rem;
        padding: 0.95rem 1rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(141, 166, 204, 0.08);
        color: #dce7f5;
        line-height: 1.5;
    }
    .shell-table {
        width: 100%;
        border-collapse: collapse;
        overflow: hidden;
        border-radius: 18px;
        margin-top: 0.9rem;
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(141, 166, 204, 0.08);
    }
    .shell-table th,
    .shell-table td {
        padding: 0.8rem 0.9rem;
        text-align: left;
        border-bottom: 1px solid rgba(141, 166, 204, 0.08);
    }
    .shell-table th {
        color: #9bb1cd;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        background: rgba(255, 255, 255, 0.03);
    }
    .shell-table td {
        color: #edf4ff;
    }
    .shell-table tr:last-child td {
        border-bottom: 0;
    }
    .result-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin-top: 1rem;
    }
    .result-metric {
        padding: 0.9rem 0.95rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(141, 166, 204, 0.08);
    }
    .result-metric span {
        display: block;
        font-size: 0.78rem;
        color: #9bb1cd;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .result-metric strong {
        font-size: 1.28rem;
        color: #edf4ff;
    }
    .soft-list {
        margin: 0.7rem 0 0;
        padding-left: 1.2rem;
        color: #9db0c8;
    }
    .stSubheader {
        color: #edf4ff !important;
    }
    .stTextArea label, .stMarkdown, .stCaption {
        color: #dce7f5 !important;
    }
    .stTextArea textarea {
        background: rgba(8, 14, 24, 0.58) !important;
        color: #edf4ff !important;
        border: 1px solid rgba(141, 166, 204, 0.14) !important;
        border-radius: 20px !important;
        min-height: 210px !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(77, 183, 214, 0.35) !important;
        box-shadow: 0 0 0 1px rgba(77, 183, 214, 0.18) !important;
    }
    .stButton > button {
        width: 100%;
        border-radius: 999px;
        border: 0;
        background: linear-gradient(135deg, #4ab8d6, #35639c);
        color: #f4fbff;
        font-weight: 700;
        padding: 0.8rem 1.1rem;
        box-shadow: 0 14px 34px rgba(53, 99, 156, 0.28);
    }
    .stButton > button:hover {
        border: 0;
        color: #ffffff;
        background: linear-gradient(135deg, #5ec5e3, #3e6faa);
    }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        border-radius: 18px;
        overflow: hidden;
    }
    .stDataFrame, .stAlert {
        border-radius: 18px;
        overflow: hidden;
    }
    div[data-baseweb="select"] > div {
        background: rgba(8, 14, 24, 0.58) !important;
        border-color: rgba(141, 166, 204, 0.14) !important;
        border-radius: 14px !important;
    }
    @media (max-width: 900px) {
        .result-grid {
            grid-template-columns: 1fr;
        }
        .hero-title {
            max-width: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <p class="hero-kicker">PYA · revisión contextual</p>
        <h1 class="hero-title">Radar de discurso y riesgo</h1>
        <p class="hero-copy">
            Esta versión usa un enfoque híbrido: combina un clasificador TF-IDF
            con reglas de contexto para estimar riesgo de odio sin tratar cualquier insulto como una confirmación automática.
            La lectura sigue siendo orientativa y debe revisarse con criterio humano.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("Analizar texto")
    analysis_view = st.selectbox(
        "Vista del análisis",
        [
            "Decisión final",
            "Modelo de regresión",
            "Reglas de contexto",
        ],
        index=0,
    )
    raw_text = st.text_area(
        "Escribe un mensaje, comentario o tweet",
        height=200,
        placeholder="Ejemplo: Esa gente no merece derechos y deberían expulsarlos.",
    )

    if st.button("Evaluar texto", use_container_width=True):
        if not raw_text.strip():
            st.warning("Escribe un texto antes de estimar el riesgo.")
        else:
            clean_text = preprocessor.clean_text(raw_text)
            result = predictor.predict_one(raw_text)
            content_type = result["content_type"]
            is_hate = content_type == "odio"
            is_abusive = content_type == "agresion_verbal"
            tag = (
                "Probabilidad alta de odio" if is_hate
                else "Agresión verbal probable" if is_abusive
                else "Sin señales fuertes de odio"
            )
            color = "#8c2f39" if is_hate else "#9a6a18" if is_abusive else "#235847"
            matched_terms = ", ".join(result["matched_terms"]) if result["matched_terms"] else "sin coincidencias claras"
            reasons = ", ".join(result["reasons"]) if result["reasons"] else "solo evidencia del modelo"
            reading = (
                "La estimación apunta a un ataque dirigido contra grupo o identidad."
                if is_hate else
                "Hay insulto directo, pero no aparecen señales suficientes de odio identitario."
                if is_abusive else
                "No hay señales claras de ataque dirigido ni de agresión fuerte."
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <h3 style="margin:0;color:{color};">{tag}</h3>
                    <p style="margin:0.55rem 0 0;color:#9eb1ca;">
                        {reading}
                    </p>
                    <div class="result-grid">
                        <div class="result-metric">
                            <span>Estimación final</span>
                            <strong>{result["confidence"]:.1%}</strong>
                        </div>
                        <div class="result-metric">
                            <span>Probabilidad modelo</span>
                            <strong>{result["model_probability"]:.1%}</strong>
                        </div>
                        <div class="result-metric">
                            <span>Ajuste por reglas</span>
                            <strong>{result["rule_score"]:.1%}</strong>
                        </div>
                    </div>
                    <p style="margin:0.9rem 0 0;"><strong>Texto limpio:</strong> {clean_text}</p>
                    <p style="margin:0.4rem 0 0;"><strong>Términos detectados:</strong> {matched_terms}</p>
                    <p style="margin:0.4rem 0 0;"><strong>Pistas usadas:</strong> {reasons}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            score_rows = [
                {
                    "label": "Regresión logística",
                    "value": result["model_probability"],
                    "caption": "Salida del clasificador TF-IDF antes de reglas.",
                    "weight": "Peso base del modelo",
                    "color": "linear-gradient(90deg, #3d8bfd, #6fb7ff)",
                },
                {
                    "label": "Reglas de contexto",
                    "value": result["rule_score"],
                    "caption": "Intensidad heurística por insultos, grupos, exclusión o violencia.",
                    "weight": "Capa heurística",
                    "color": "linear-gradient(90deg, #46c2c6, #74f0d0)",
                },
                {
                    "label": "Decisión final",
                    "value": result["probabilities"]["odio"],
                    "caption": "Combinación ponderada 70% modelo y 30% reglas.",
                    "weight": "Score de riesgo final",
                    "color": "linear-gradient(90deg, #ff7d66, #ffb36b)" if is_hate else "linear-gradient(90deg, #57b9f7, #56d0bd)",
                },
            ]

            if analysis_view == "Modelo de regresión":
                score_rows = [score_rows[0], score_rows[2]]
                explanation = "La vista del modelo muestra la probabilidad que aprende la regresión logística sobre vectores TF-IDF y cómo termina impactando en el score combinado."
            elif analysis_view == "Reglas de contexto":
                score_rows = [score_rows[1], score_rows[2]]
                explanation = "La vista de reglas enfatiza señales directas: grupos objetivo, exclusión, violencia, insulto y contexto de ataque."
            else:
                explanation = "La decisión final contrasta el aporte del modelo estadístico con el ajuste de reglas para que no parezca una caja negra."

            score_markup = "".join(
                f"""
                <div class="score-track">
                    <div class="weight-chip">{row["weight"]}</div>
                    <header>
                        <span>{row["label"]}</span>
                        <strong>{row["value"]:.1%}</strong>
                    </header>
                    <div class="score-bar">
                        <div class="score-fill" style="width:{row["value"] * 100:.1f}%;background:{row["color"]};"></div>
                    </div>
                    <small>{row["caption"]}</small>
                </div>
                """
                for row in score_rows
            )

            table_rows = [
                ("No odio", f'{result["probabilities"]["no_odio"]:.1%}'),
                ("Odio", f'{result["probabilities"]["odio"]:.1%}'),
                ("Tipo de contenido", content_type),
                (
                    "Grupos objetivo",
                    ", ".join(result["target_groups"]) if result["target_groups"] else "sin grupo objetivo",
                ),
            ]
            table_markup = "".join(
                f"<tr><td>{label}</td><td>{value}</td></tr>"
                for label, value in table_rows
            )

            st.markdown(
                f"""
                <div class="scoreboard">
                    {score_markup}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="shell-note">
                    <strong>Lectura de esta vista:</strong> {explanation}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <table class="shell-table">
                    <thead>
                        <tr>
                            <th>Componente</th>
                            <th>Valor</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_markup}
                    </tbody>
                </table>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <ul class="soft-list">
                <li>Prioriza ataques dirigidos a grupos o identidades.</li>
                <li>Baja el peso de quejas comunes, crítica política o insultos no identitarios.</li>
                <li>Muestra una estimación probabilística separando modelo y reglas.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

with right_col:
    st.subheader("Ejemplos del corpus")
    twitter_path = DATA_DIR / "twitter_posts.csv"

    if twitter_path.exists():
        tweets_df = pd.read_csv(twitter_path)
        preview_df = tweets_df.copy()
        if "clean_text" not in preview_df.columns and "text" in preview_df.columns:
            preview_df["clean_text"] = preview_df["text"].astype(str).apply(preprocessor.clean_text)

        preview_df["prediccion"] = preview_df["clean_text"].apply(
            lambda text: predictor.predict_one(str(text))["label"]
        )
        preview_df["score_odio"] = preview_df["clean_text"].apply(
            lambda text: predictor.predict_one(str(text))["probabilities"]["odio"]
        )
        st.dataframe(
            preview_df[["text", "prediccion", "score_odio"]].head(20),
            use_container_width=True,
        )
    else:
        demo_rows = pd.DataFrame(
            {
                "text": [
                    "No estoy de acuerdo con tu propuesta, pero podemos debatirla.",
                    "Esa gente no merece derechos y deberían expulsarlos.",
                    "Pinches indios, nunca aprenden.",
                    "La película estuvo horrible y perdí mi dinero.",
                    "m*tate",
                ]
            }
        )
        demo_rows["clean_text"] = demo_rows["text"].astype(str).apply(preprocessor.clean_text)
        demo_rows["prediccion"] = demo_rows["text"].apply(
            lambda text: predictor.predict_one(str(text))["label"]
        )
        demo_rows["score_odio"] = demo_rows["text"].apply(
            lambda text: predictor.predict_one(str(text))["probabilities"]["odio"]
        )
        st.dataframe(
            demo_rows[["text", "prediccion", "score_odio"]],
            use_container_width=True,
        )
        st.caption("Vista de respaldo con ejemplos locales. Si luego existe data/twitter_posts.csv, este panel mostrará esos registros.")

st.caption("Frontend local con Streamlit usando una arquitectura híbrida: clasificador TF-IDF más reglas de contexto. Su lectura es orientativa, no definitiva.")
