import csv
import html
import io
import json
import time

import pandas as pd
import streamlit as st
import torch
from transformers import pipeline


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_ID = "AbdelrahmanAkl/NewsNER-DistilBERT"

LABELS = {
    "PER": "Person",
    "ORG": "Organization",
    "LOC": "Location",
    "MISC": "Miscellaneous",
}

# label -> (accent, soft background, border)
COLORS = {
    "PER": ("#3B4BDB", "#EDF0FF", "#C5CCFF"),
    "ORG": ("#0F8A5F", "#E6F6EF", "#B5E3CF"),
    "LOC": ("#C2640A", "#FFF3E0", "#F7D4A3"),
    "MISC": ("#B8265F", "#FDEBF2", "#F5C2D6"),
}

SAMPLES = {
    "Tech announcement": (
        "Apple announced a new artificial intelligence research center in London. "
        "The company said that John Smith, Apple's Vice President of AI, "
        "will lead the new team. Microsoft and Google are also expected to "
        "participate in the European AI initiative."
    ),
    "World news": (
        "German Chancellor Angela Merkel met French President Emmanuel Macron in Berlin "
        "on Tuesday to discuss the European Union budget. The talks were also attended "
        "by officials from the European Commission and the International Monetary Fund."
    ),
    "Sports": (
        "Liverpool beat Manchester City 3-1 at Anfield on Sunday, with Mohamed Salah "
        "scoring twice. The result keeps Liverpool at the top of the Premier League table."
    ),
}


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="NewsNER-AI",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="st-"], .stMarkdown { font-family: 'Inter', system-ui, sans-serif; }
#MainMenu, footer { visibility: hidden; }

.block-container { max-width: 1120px; padding-top: 2.4rem; padding-bottom: 3rem; }

.hero h1 {
    font-size: 2.6rem; font-weight: 700; letter-spacing: -0.04em;
    color: #14171F; margin: 0; line-height: 1.1;
}
.hero p { color: #5B6272; font-size: 1.05rem; margin: .6rem 0 0; max-width: 640px; line-height: 1.6; }

.chips { display: flex; flex-wrap: wrap; gap: .5rem; margin: 1.1rem 0 .2rem; }
.chip {
    background: #fff; border: 1px solid #E3E6EC; border-radius: 999px;
    padding: .28rem .8rem; font-size: .82rem; color: #3A4152;
}
.chip b { color: #14171F; font-weight: 600; }

.section-title { color: #14171F; font-size: 1.15rem; font-weight: 650; margin: 2rem 0 .8rem; }

.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .8rem; }
.stat {
    background: #fff; border: 1px solid #E3E6EC; border-radius: 12px;
    padding: .9rem 1.05rem; border-left-width: 4px;
}
.stat .n { font-size: 1.9rem; font-weight: 700; line-height: 1.1; color: #14171F; }
.stat .l { font-size: .85rem; color: #5B6272; margin-top: .15rem; }

.ent-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: .8rem; }
.ent-card {
    background: #fff; border: 1px solid #E3E6EC; border-radius: 12px; padding: .95rem 1.05rem;
}
.ent-top { display: flex; justify-content: space-between; align-items: center; gap: .5rem; }
.ent-type { font-size: .78rem; font-weight: 600; padding: .12rem .55rem; border-radius: 999px; border: 1px solid; }
.ent-count { font-size: .78rem; color: #5B6272; }
.ent-text { color: #14171F; font-size: 1.08rem; font-weight: 600; margin-top: .55rem; word-break: break-word; }
.bar { height: 5px; border-radius: 99px; background: #EEF0F4; margin-top: .7rem; overflow: hidden; }
.bar > div { height: 100%; border-radius: 99px; }
.ent-conf { color: #5B6272; font-size: .8rem; margin-top: .35rem; }

.article-box {
    background: #fff; border: 1px solid #E3E6EC; border-radius: 12px;
    padding: 1.4rem 1.5rem; color: #2B3142; font-size: 1.02rem; line-height: 2.1;
}
.ent {
    border: 1px solid; border-radius: 6px; padding: 1px 6px; margin: 0 1px;
    font-weight: 600; color: #14171F; white-space: nowrap;
}
.ent .tag { font-size: .68rem; font-weight: 700; margin-left: 5px; }

.legend { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: .8rem; font-size: .84rem; color: #3A4152; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; }

.footer { text-align: center; color: #8A91A0; font-size: .82rem; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #E8EAEF; }

.stButton > button { border-radius: 10px; font-weight: 600; min-height: 2.75rem; }
textarea { border-radius: 12px !important; font-size: 1rem !important; line-height: 1.6 !important; }

@media (max-width: 760px) { .stat-grid { grid-template-columns: repeat(2, 1fr); } }
</style>
""",
    unsafe_allow_html=True,
)


def render(markup: str):
    """Render HTML without Streamlit treating indented lines as a code block."""
    flat = "".join(line.strip() for line in markup.splitlines())
    st.markdown(flat, unsafe_allow_html=True)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_ner_model():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        task="token-classification",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        aggregation_strategy="simple",
        stride=64,  # lets long articles be processed in overlapping windows
        device=device,
    )


def run_ner(text: str):
    ner = load_ner_model()
    start = time.perf_counter()
    raw = ner(text)
    elapsed_ms = (time.perf_counter() - start) * 1000

    entities = []
    for item in raw:
        s, e = item.get("start"), item.get("end")
        label = item.get("entity_group")
        if s is None or e is None or label not in LABELS:
            continue
        entities.append(
            {
                "text": text[int(s):int(e)].strip(),
                "label": label,
                "score": float(item.get("score", 0.0)),
                "start": int(s),
                "end": int(e),
            }
        )
    entities.sort(key=lambda x: x["start"])
    return entities, elapsed_ms


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def highlight_article(text, entities):
    out, cursor = [], 0
    for ent in entities:
        if ent["start"] < cursor:
            continue
        accent, soft, border = COLORS[ent["label"]]
        out.append(html.escape(text[cursor:ent["start"]]))
        out.append(
            f'<span class="ent" style="background:{soft};border-color:{border}" '
            f'title="{html.escape(LABELS[ent["label"]])} - {ent["score"]:.1%}">'
            f'{html.escape(text[ent["start"]:ent["end"]])}'
            f'<span class="tag" style="color:{accent}">{ent["label"]}</span></span>'
        )
        cursor = ent["end"]
    out.append(html.escape(text[cursor:]))
    return "".join(out).replace("\n", "<br>")


def group_entities(entities):
    groups = {}
    for ent in entities:
        key = (ent["text"].lower(), ent["label"])
        g = groups.setdefault(
            key, {"text": ent["text"], "label": ent["label"], "mentions": 0, "score": 0.0}
        )
        g["mentions"] += 1
        g["score"] = max(g["score"], ent["score"])
    return sorted(groups.values(), key=lambda g: (-g["mentions"], -g["score"]))


def set_sample():
    choice = st.session_state.get("sample_choice")
    if choice in SAMPLES:
        st.session_state["article_text"] = SAMPLES[choice]
        st.session_state["result"] = None


def clear_all():
    st.session_state["article_text"] = ""
    st.session_state["result"] = None
    st.session_state["sample_choice"] = "Choose an example"


st.session_state.setdefault("article_text", SAMPLES["Tech announcement"])
st.session_state.setdefault("result", None)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

render(
    f"""
<div class="hero">
  <h1>NewsNER-AI</h1>
  <p>Paste a news article and instantly see the people, organizations,
  locations and other named entities it mentions.</p>
</div>
<div class="chips">
  <span class="chip">Model: <b>DistilBERT (fine-tuned)</b></span>
  <span class="chip">Trained on: <b>CoNLL-2003</b></span>
  <span class="chip">Runs on: <b>{"GPU" if torch.cuda.is_available() else "CPU"}</b></span>
</div>
"""
)


# ---------------------------------------------------------
# Input
# ---------------------------------------------------------

render('<div class="section-title">Your article</div>')

st.selectbox(
    "Example",
    ["Choose an example", *SAMPLES.keys()],
    key="sample_choice",
    on_change=set_sample,
    label_visibility="collapsed",
)

st.text_area(
    "News article",
    key="article_text",
    height=200,
    placeholder="Paste a news article here...",
    label_visibility="collapsed",
)

c1, c2 = st.columns([3, 1])
with c1:
    analyze = st.button("Find entities", type="primary", use_container_width=True)
with c2:
    st.button("Clear", on_click=clear_all, use_container_width=True)

if analyze:
    text = st.session_state["article_text"]
    if not text.strip():
        st.warning("Paste an article first, then select Find entities.")
    else:
        with st.spinner("Reading the article..."):
            entities, ms = run_ner(text)
        st.session_state["result"] = {"text": text, "entities": entities, "ms": ms}


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

result = st.session_state["result"]

if result:
    text, all_entities = result["text"], result["entities"]

    render('<div class="section-title">Results</div>')

    f1, f2 = st.columns([1, 2])
    with f1:
        threshold = st.slider(
            "Minimum confidence", 0.0, 1.0, 0.5, 0.05, format="%.2f",
            help="Hide entities the model is less sure about.",
        )
    with f2:
        selected = st.multiselect(
            "Entity types",
            options=list(LABELS),
            default=list(LABELS),
            format_func=lambda k: LABELS[k],
        )

    entities = [e for e in all_entities if e["score"] >= threshold and e["label"] in selected]

    if not entities:
        st.info("No entities match these filters. Lower the confidence or add more entity types.")
    else:
        counts = {k: sum(1 for e in entities if e["label"] == k) for k in LABELS}
        render(
            '<div class="stat-grid">'
            + "".join(
                f'<div class="stat" style="border-left-color:{COLORS[k][0]}">'
                f'<div class="n">{counts[k]}</div><div class="l">{LABELS[k]}</div></div>'
                for k in LABELS
            )
            + "</div>"
        )
        st.caption(f"{len(entities)} mentions found in {result['ms']:.0f} ms")

        tab_text, tab_cards, tab_table = st.tabs(["Highlighted article", "Entities", "Table and export"])

        with tab_text:
            render(
                '<div class="legend">'
                + "".join(
                    f'<span><i style="background:{COLORS[k][1]};border:1px solid {COLORS[k][2]}"></i>{LABELS[k]}</span>'
                    for k in LABELS
                )
                + "</div>"
            )
            render(f'<div class="article-box">{highlight_article(text, entities)}</div>')

        grouped = group_entities(entities)

        with tab_cards:
            cards = []
            for g in grouped:
                accent, soft, border = COLORS[g["label"]]
                mentions = f'{g["mentions"]} mentions' if g["mentions"] > 1 else "1 mention"
                cards.append(
                    f'<div class="ent-card">'
                    f'<div class="ent-top">'
                    f'<span class="ent-type" style="color:{accent};background:{soft};border-color:{border}">{LABELS[g["label"]]}</span>'
                    f'<span class="ent-count">{mentions}</span></div>'
                    f'<div class="ent-text">{html.escape(g["text"])}</div>'
                    f'<div class="bar"><div style="width:{g["score"]*100:.1f}%;background:{accent}"></div></div>'
                    f'<div class="ent-conf">{g["score"]:.1%} confidence</div>'
                    f"</div>"
                )
            render('<div class="ent-grid">' + "".join(cards) + "</div>")

        with tab_table:
            df = pd.DataFrame(
                [
                    {
                        "Entity": g["text"],
                        "Type": LABELS[g["label"]],
                        "Mentions": g["mentions"],
                        "Confidence": g["score"] * 100,
                    }
                    for g in grouped
                ]
            )
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence", min_value=0, max_value=100, format="%.1f%%"
                    )
                },
            )

            payload = [
                {"text": e["text"], "type": e["label"], "confidence": round(e["score"], 4),
                 "start": e["start"], "end": e["end"]}
                for e in entities
            ]
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=list(payload[0]))
            writer.writeheader()
            writer.writerows(payload)

            d1, d2, _ = st.columns([1, 1, 2])
            with d1:
                st.download_button(
                    "Download JSON", json.dumps(payload, indent=2, ensure_ascii=False),
                    file_name="entities.json", mime="application/json", use_container_width=True,
                )
            with d2:
                st.download_button(
                    "Download CSV", buf.getvalue(),
                    file_name="entities.csv", mime="text/csv", use_container_width=True,
                )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

render('<div class="footer">NewsNER-AI. A DistilBERT model fine-tuned on CoNLL-2003.</div>')
