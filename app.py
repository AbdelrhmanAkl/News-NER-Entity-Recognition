import html

import streamlit as st
import torch
from transformers import pipeline


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_ID = "AbdelrahmanAkl/NewsNER-DistilBERT"

ENTITY_LABELS = {
    "PER": "Person",
    "ORG": "Organization",
    "LOC": "Location",
    "MISC": "Miscellaneous",
}

DEFAULT_TEXT = (
    "Apple announced a new artificial intelligence research center in London. "
    "The company said that John Smith, Apple's Vice President of AI, "
    "will lead the new team. Microsoft and Google are also expected to "
    "participate in the European AI initiative."
)


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="NewsNER-AI",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 1rem 0 1.2rem 0;
        }

        .hero h1 {
            font-size: 2.55rem;
            font-weight: 750;
            letter-spacing: -0.035em;
            color: #101828;
            margin: 0;
            line-height: 1.15;
        }

        .hero p {
            color: #667085;
            font-size: 1.02rem;
            margin: 0.55rem 0 0 0;
            max-width: 720px;
            line-height: 1.6;
        }

        .section-title {
            color: #101828;
            font-size: 1.2rem;
            font-weight: 700;
            margin-top: 1.6rem;
            margin-bottom: 0.8rem;
        }

        .entity-card {
            background: #ffffff;
            border: 1px solid #e4e7ec;
            border-radius: 12px;
            padding: 1rem 1.05rem;
            margin-bottom: 0.8rem;
            transition: border-color 0.15s ease;
        }

        .entity-card:hover {
            border-color: #cbd5e1;
        }

        .entity-type {
            color: #667085;
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.065em;
            text-transform: uppercase;
        }

        .entity-text {
            color: #101828;
            font-size: 1.04rem;
            font-weight: 650;
            margin-top: 0.25rem;
            word-break: break-word;
        }

        .entity-confidence {
            color: #667085;
            font-size: 0.82rem;
            margin-top: 0.28rem;
        }

        .article-box {
            background: #ffffff;
            border: 1px solid #e4e7ec;
            border-radius: 12px;
            padding: 1.35rem 1.45rem;
            color: #344054;
            font-size: 1rem;
            line-height: 1.95;
            white-space: normal;
        }

        .entity-highlight {
            background: #eef2ff;
            border: 1px solid #c7d2fe;
            border-radius: 5px;
            padding: 2px 5px;
            color: #1e293b;
            font-weight: 650;
        }

        .model-note {
            color: #667085;
            font-size: 0.82rem;
            margin-top: 0.45rem;
        }

        .footer {
            text-align: center;
            color: #98a2b3;
            font-size: 0.82rem;
            margin-top: 3.2rem;
            padding-top: 1rem;
            border-top: 1px solid #f2f4f7;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e4e7ec;
            border-radius: 12px;
            padding: 0.9rem 1rem;
        }

        div[data-testid="stMetricLabel"] {
            color: #667085;
        }

        div[data-testid="stMetricValue"] {
            color: #101828;
        }

        .stButton > button {
            border-radius: 8px;
            font-weight: 650;
            min-height: 2.7rem;
        }

        textarea {
            border-radius: 10px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_ner_model():
    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        task="token-classification",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        aggregation_strategy="simple",
        device=device,
    )


# ---------------------------------------------------------
# Entity utilities
# ---------------------------------------------------------

def build_highlighted_article(text, entities):
    """
    Highlight detected entities while preserving the original article text.
    """

    if not entities:
        return html.escape(text)

    spans = []

    for entity in entities:
        start = entity.get("start")
        end = entity.get("end")
        label = entity.get("entity_group")

        if start is None or end is None or not label:
            continue

        spans.append(
            (
                int(start),
                int(end),
                label,
            )
        )

    spans.sort(key=lambda item: item[0])

    result = []
    cursor = 0

    for start, end, label in spans:
        if start < cursor:
            continue

        result.append(
            html.escape(text[cursor:start])
        )

        entity_text = html.escape(
            text[start:end]
        )

        safe_label = html.escape(label)

        result.append(
            f'<span class="entity-highlight" '
            f'title="{html.escape(ENTITY_LABELS.get(label, label))}">'
            f"{entity_text} [{safe_label}]"
            f"</span>"
        )

        cursor = end

    result.append(
        html.escape(text[cursor:])
    )

    return "".join(result)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>NewsNER-AI</h1>
        <p>
            Named Entity Recognition for news articles powered by a
            fine-tuned DistilBERT model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Model overview
# ---------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "DistilBERT",
    )

with col2:
    st.metric(
        "Entity Types",
        "4",
    )

with col3:
    st.metric(
        "Inference",
        "GPU" if torch.cuda.is_available() else "CPU",
    )

st.markdown(
    """
    <div class="model-note">
        Fine-tuned on the CoNLL-2003 news dataset for PER, ORG, LOC and MISC entities.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Input section
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Analyze a News Article</div>',
    unsafe_allow_html=True,
)

article = st.text_area(
    "News article",
    value=DEFAULT_TEXT,
    height=230,
    placeholder="Paste a news article here...",
    label_visibility="collapsed",
)

button_col1, button_col2 = st.columns([3, 1])

with button_col1:
    analyze = st.button(
        "Analyze Article",
        type="primary",
        use_container_width=True,
    )

with button_col2:
    clear = st.button(
        "Clear",
        use_container_width=True,
    )

if clear:
    st.session_state["article"] = ""
    st.rerun()


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

if analyze:
    if not article.strip():
        st.warning(
            "Please enter a news article before running the analysis."
        )
        st.stop()

    with st.spinner("Analyzing article..."):
        ner = load_ner_model()
        entities = ner(article)

    entities = sorted(
        entities,
        key=lambda entity: entity.get("start", 0),
    )

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Detected Entities</div>',
        unsafe_allow_html=True,
    )

    if not entities:
        st.info(
            "No named entities were detected in this article."
        )
        st.stop()

    counts = {
        "PER": 0,
        "ORG": 0,
        "LOC": 0,
        "MISC": 0,
    }

    for entity in entities:
        label = entity.get("entity_group")

        if label in counts:
            counts[label] += 1

    metric_columns = st.columns(4)

    for column, label in zip(
        metric_columns,
        ["PER", "ORG", "LOC", "MISC"],
    ):
        with column:
            st.metric(
                ENTITY_LABELS[label],
                counts[label],
            )

    st.markdown("")

    entity_columns = st.columns(2)

    for index, entity in enumerate(entities):
        column = entity_columns[index % 2]

        label = entity.get("entity_group", "UNKNOWN")
        entity_text = entity.get("word", "")
        score = float(entity.get("score", 0.0))

        with column:
            st.markdown(
                f"""
                <div class="entity-card">
                    <div class="entity-type">
                        {html.escape(ENTITY_LABELS.get(label, label))}
                    </div>
                    <div class="entity-text">
                        {html.escape(entity_text)}
                    </div>
                    <div class="entity-confidence">
                        {score:.2%} confidence
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Entity-Highlighted Article</div>',
        unsafe_allow_html=True,
    )

    highlighted_article = build_highlighted_article(
        article,
        entities,
    )

    st.markdown(
        f"""
        <div class="article-box">
            {highlighted_article}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        NewsNER-AI · CoNLL-2003 · Fine-tuned DistilBERT
    </div>
    """,
    unsafe_allow_html=True,
)