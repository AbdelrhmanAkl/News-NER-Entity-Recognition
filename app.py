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
MODEL_URL = f"https://huggingface.co/{MODEL_ID}"

# label -> (singular, plural, accent, soft tint)
ENTITY = {
    "PER": ("Person", "People", "#8EA2FF", "rgba(142,162,255,.18)"),
    "ORG": ("Organization", "Organizations", "#4FD8B0", "rgba(79,216,176,.16)"),
    "LOC": ("Location", "Locations", "#FFB86B", "rgba(255,184,107,.17)"),
    "MISC": ("Other name", "Other names", "#FF8FB5", "rgba(255,143,181,.17)"),
}
SHORT = {"PER": "person", "ORG": "org", "LOC": "place", "MISC": "other"}

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
# Page setup and styling
# ---------------------------------------------------------

st.set_page_config(
    page_title="NewsNER-AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&display=swap');

:root {
    --bg: #0B1020; --surface: #121A30; --line: #243050;
    --text: #E8ECF6; --muted: #8D99B5; --accent: #8EA2FF;
    --serif: 'Newsreader', Georgia, serif;
}

html, body, .stApp, [class*="st-"] { font-family: 'Instrument Sans', system-ui, sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1120px; padding-top: 1.4rem; padding-bottom: 3rem; }

/* top bar */
.nav { display: flex; justify-content: space-between; align-items: center; padding-bottom: 1rem; border-bottom: 1px solid var(--line); }
.brand { display: flex; align-items: center; gap: .6rem; font-weight: 600; font-size: 1.05rem; letter-spacing: -0.01em; color: var(--text); }
.status { font-size: .84rem; color: var(--muted); display: flex; align-items: center; gap: .5rem; }
.status i { width: 8px; height: 8px; border-radius: 50%; background: #4FD8B0; display: inline-block; }
.status a { color: var(--muted); text-decoration: underline; text-underline-offset: 3px; }

/* hero */
.hero { padding: 3.2rem 0 1.6rem; }
.hero h1 {
    font-family: var(--serif); font-weight: 500; color: var(--text);
    font-size: clamp(2.3rem, 5.2vw, 3.9rem); line-height: 1.04; letter-spacing: -0.025em;
    max-width: 15ch; margin: 0;
}
.hero p { color: var(--muted); font-size: 1.08rem; line-height: 1.65; max-width: 560px; margin: 1.1rem 0 0; }

/* inputs */
textarea {
    background: var(--surface) !important; color: var(--text) !important;
    border: 1px solid var(--line) !important; border-radius: 14px !important;
    font-family: var(--serif) !important; font-size: 1.12rem !important; line-height: 1.7 !important;
    padding: 1rem 1.15rem !important;
}
textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 1px var(--accent) !important; }
[data-baseweb="select"] > div { background: var(--surface) !important; border-color: var(--line) !important; color: var(--text) !important; }
span[data-baseweb="tag"] { background: rgba(142,162,255,.18) !important; color: var(--text) !important; }
.stButton > button { border-radius: 12px; font-weight: 600; min-height: 2.8rem; }
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: var(--accent); color: #0B1020; border: 0;
}
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover { background: #A9B8FF; color: #0B1020; }
.stButton > button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
    background: transparent; color: var(--text); border: 1px solid var(--line);
}
div[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: transparent; }

/* section headings */
.kicker { font-family: var(--serif); font-size: 1.5rem; font-weight: 500; letter-spacing: -0.01em; margin: 2.4rem 0 .9rem; color: var(--text); }
.sub { color: var(--muted); font-size: .9rem; margin: -.5rem 0 1rem; }

/* distribution bar */
.dist { display: flex; gap: 4px; height: 10px; margin: .4rem 0 .7rem; }
.dist div { border-radius: 99px; min-width: 10px; }
.dist-legend { display: flex; flex-wrap: wrap; gap: 1.3rem; font-size: .9rem; color: var(--muted); margin-bottom: 1.6rem; }
.dist-legend b { color: var(--text); font-weight: 600; margin-left: .3rem; }
.dist-legend i { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: .45rem; }

/* annotated article */
.paper {
    background: var(--surface); border: 1px solid var(--line); border-radius: 16px;
    padding: 1.9rem 2rem; font-family: var(--serif); font-size: 1.32rem; line-height: 2.15; color: #D3DAEA;
}
.ent {
    --c: #fff; --soft: transparent; --i: 0;
    color: var(--text); font-weight: 500; padding: 1px 4px 2px; border-radius: 4px;
    border-bottom: 2px solid var(--c);
    background-image: linear-gradient(var(--soft), var(--soft));
    background-repeat: no-repeat; background-size: 0% 100%;
    animation: sweep .7s cubic-bezier(.2,.7,.2,1) forwards;
    animation-delay: calc(var(--i) * 120ms + 150ms);
}
.ent sup { font-family: 'Instrument Sans', sans-serif; font-size: .62rem; font-weight: 600; margin-left: 4px; color: var(--c); letter-spacing: .02em; }
@keyframes sweep { to { background-size: 100% 100%; } }
@media (prefers-reduced-motion: reduce) { .ent { animation: none; background-size: 100% 100%; } }

/* index */
.idx { border-left: 1px solid var(--line); padding-left: 1.4rem; }
.idx h4 { display: flex; align-items: center; gap: .55rem; font-size: .95rem; font-weight: 600; color: var(--text); margin: 0 0 .5rem; }
.idx h4 i { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.idx h4 span { color: var(--muted); font-weight: 500; margin-left: auto; }
.idx ul { list-style: none; padding: 0; margin: 0 0 1.5rem; }
.idx li { padding: .5rem 0; border-top: 1px solid var(--line); }
.idx .row { display: flex; justify-content: space-between; gap: .8rem; align-items: baseline; }
.idx .nm { color: var(--text); font-weight: 500; word-break: break-word; }
.idx .meta { color: var(--muted); font-size: .8rem; white-space: nowrap; }
.idx .mini { height: 3px; background: #1B2542; border-radius: 99px; margin-top: .45rem; overflow: hidden; }
.idx .mini div { height: 100%; border-radius: 99px; }

.foot { margin-top: 3.5rem; padding-top: 1.2rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .85rem; display: flex; justify-content: space-between; flex-wrap: wrap; gap: .5rem; }
.foot a { color: var(--muted); text-underline-offset: 3px; }
</style>
""",
    unsafe_allow_html=True,
)


def render(markup: str):
    """Render HTML without Streamlit treating indented lines as a code block."""
    flat = " ".join(line.strip() for line in markup.splitlines() if line.strip())
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
        stride=64,  # long articles are read in overlapping windows
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
        if s is None or e is None or label not in ENTITY:
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

def annotate(text, entities):
    out, cursor, i = [], 0, 0
    for ent in entities:
        if ent["start"] < cursor:
            continue
        singular, _, accent, soft = ENTITY[ent["label"]]
        out.append(html.escape(text[cursor:ent["start"]]))
        out.append(
            f'<span class="ent" style="--c:{accent};--soft:{soft};--i:{i}" '
            f'title="{html.escape(singular)}, {ent["score"]:.1%} confidence">'
            f'{html.escape(text[ent["start"]:ent["end"]])}'
            f'<sup>{SHORT[ent["label"]]}</sup></span>'
        )
        cursor = ent["end"]
        i += 1
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


def build_index(grouped):
    parts = []
    for label, (_, plural, accent, _) in ENTITY.items():
        items = [g for g in grouped if g["label"] == label]
        if not items:
            continue
        rows = []
        for g in items:
            extra = f'{g["mentions"]} mentions, ' if g["mentions"] > 1 else ""
            rows.append(
                f'<li><div class="row"><span class="nm">{html.escape(g["text"])}</span>'
                f'<span class="meta">{extra}{g["score"]:.0%}</span></div>'
                f'<div class="mini"><div style="width:{g["score"]*100:.1f}%;background:{accent}"></div></div></li>'
            )
        parts.append(
            f'<h4><i style="background:{accent}"></i>{plural}<span>{len(items)}</span></h4>'
            f'<ul>{"".join(rows)}</ul>'
        )
    return '<div class="idx">' + "".join(parts) + "</div>"


def set_sample():
    choice = st.session_state.get("sample_choice")
    if choice in SAMPLES:
        st.session_state["article_text"] = SAMPLES[choice]
        st.session_state["result"] = None


def clear_all():
    st.session_state["article_text"] = ""
    st.session_state["result"] = None
    st.session_state["sample_choice"] = "Try an example"


st.session_state.setdefault("article_text", SAMPLES["Tech announcement"])
st.session_state.setdefault("result", None)


# ---------------------------------------------------------
# Top bar and hero
# ---------------------------------------------------------

device = "GPU" if torch.cuda.is_available() else "CPU"

render(
    f"""
<div class="nav">
  <div class="brand">
    <svg width="26" height="26" viewBox="0 0 26 26" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M8 4H4v18h4M18 4h4v18h-4" stroke="#8EA2FF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="13" cy="13" r="3" fill="#4FD8B0"/>
    </svg>
    NewsNER-AI
  </div>
  <div class="status"><i></i>
    <a href="{MODEL_URL}" target="_blank">DistilBERT</a> running on {device}
  </div>
</div>
<div class="hero">
  <h1>See who and what a story is about.</h1>
  <p>Paste a news article. NewsNER-AI marks every person, organization and place it
  names, and shows how sure it is about each one.</p>
</div>
"""
)


# ---------------------------------------------------------
# Input
# ---------------------------------------------------------

st.selectbox(
    "Example",
    ["Try an example", *SAMPLES.keys()],
    key="sample_choice",
    on_change=set_sample,
    label_visibility="collapsed",
)

st.text_area(
    "News article",
    key="article_text",
    height=190,
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

    with st.expander("Filters"):
        f1, f2 = st.columns([1, 2])
        with f1:
            threshold = st.slider(
                "Minimum confidence", 0.0, 1.0, 0.5, 0.05, format="%.2f",
                help="Hide entities the model is less sure about.",
            )
        with f2:
            selected = st.multiselect(
                "Entity types",
                options=list(ENTITY),
                default=list(ENTITY),
                format_func=lambda k: ENTITY[k][0],
            )

    entities = [e for e in all_entities if e["score"] >= threshold and e["label"] in selected]

    if not entities:
        st.info("No entities match these filters. Lower the confidence or add more entity types.")
    else:
        counts = {k: sum(1 for e in entities if e["label"] == k) for k in ENTITY}
        present = [k for k in ENTITY if counts[k]]

        render(f'<div class="kicker">{len(entities)} entities found in {result["ms"]:.0f} ms</div>')
        render(
            '<div class="dist">'
            + "".join(f'<div style="flex:{counts[k]};background:{ENTITY[k][2]}"></div>' for k in present)
            + "</div>"
            + '<div class="dist-legend">'
            + "".join(
                f'<span><i style="background:{ENTITY[k][2]}"></i>{ENTITY[k][1]}<b>{counts[k]}</b></span>'
                for k in present
            )
            + "</div>"
        )

        grouped = group_entities(entities)

        left, right = st.columns([5, 3], gap="large")
        with left:
            render(f'<div class="paper">{annotate(text, entities)}</div>')
        with right:
            render(build_index(grouped))

        with st.expander("Table and export"):
            df = pd.DataFrame(
                [
                    {
                        "Entity": g["text"],
                        "Type": ENTITY[g["label"]][0],
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

render(
    f"""
<div class="foot">
  <span>NewsNER-AI. DistilBERT fine-tuned on CoNLL-2003.</span>
  <span><a href="{MODEL_URL}" target="_blank">Model on Hugging Face</a></span>
</div>
"""
)
