import csv
import html
import io
import json
import time

import streamlit as st
import torch
from transformers import pipeline


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_ID = "AbdelrahmanAkl/NewsNER-DistilBERT"
MODEL_URL = f"https://huggingface.co/{MODEL_ID}"
MIN_CONFIDENCE = 0.50  # entities below this score are hidden

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
    --sans: 'Instrument Sans', system-ui, sans-serif;
}

/* base: everything is styled here, so the page looks the same with or without a theme file */
.stApp, [data-testid="stAppViewContainer"] {
    background: radial-gradient(900px 380px at 12% -8%, rgba(142,162,255,.13), transparent 60%), var(--bg);
    color: var(--text);
}
.stApp, .stApp p, .stApp label, .stApp button, .stApp textarea, .stApp input { font-family: var(--sans); }
[data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}
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
.hero { padding: 3rem 0 1.4rem; }
.headline {
    font-family: var(--serif) !important; font-weight: 500; color: var(--text);
    font-size: clamp(2.4rem, 5.4vw, 4rem); line-height: 1.04; letter-spacing: -0.025em; max-width: 15ch;
}
.lede { color: var(--muted); font-size: 1.08rem; line-height: 1.65; max-width: 560px; margin-top: 1.1rem; }

/* example chips */
.try { color: var(--muted); font-size: .88rem; margin: 0 0 .5rem; }
[class*="st-key-ex_"] button {
    min-height: 2.1rem; padding: 0 .9rem; font-size: .86rem; font-weight: 500;
    border-radius: 999px; background: transparent; color: var(--muted); border: 1px solid var(--line);
}
[class*="st-key-ex_"] button:hover { color: var(--text); border-color: var(--accent); background: rgba(142,162,255,.08); }

/* text area */
div[data-baseweb="textarea"], div[data-baseweb="base-input"] {
    background: var(--surface) !important; border: 1px solid var(--line) !important; border-radius: 14px !important;
}
div[data-baseweb="textarea"]:focus-within { border-color: var(--accent) !important; box-shadow: 0 0 0 1px var(--accent); }
textarea {
    background: transparent !important; color: var(--text) !important;
    font-family: var(--serif) !important; font-size: 1.12rem !important; line-height: 1.7 !important;
    padding: 1rem 1.15rem !important;
}

/* buttons */
.stButton > button, .stDownloadButton > button { border-radius: 12px; font-weight: 600; min-height: 2.8rem; }
button[data-testid="stBaseButton-primary"], .stButton > button[kind="primary"] { background: var(--accent); color: #0B1020; border: 0; }
button[data-testid="stBaseButton-primary"]:hover, .stButton > button[kind="primary"]:hover { background: #A9B8FF; color: #0B1020; }
button[data-testid="stBaseButton-secondary"], .stButton > button[kind="secondary"], .stDownloadButton > button {
    background: transparent; color: var(--text); border: 1px solid var(--line);
}
button[data-testid="stBaseButton-secondary"]:hover, .stDownloadButton > button:hover { border-color: var(--accent); color: var(--text); }
[data-testid="stSpinner"], [data-testid="stSpinner"] * { color: var(--muted) !important; }

.notice { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: .9rem 1.1rem; color: var(--muted); margin-top: 1rem; }

/* result header */
.kicker { font-family: var(--serif); font-size: 1.6rem; font-weight: 500; letter-spacing: -0.01em; margin: 2.6rem 0 .9rem; color: var(--text); }
.dist { display: flex; gap: 4px; height: 10px; margin: .4rem 0 .7rem; }
.dist div { border-radius: 99px; min-width: 10px; }
.dist-legend { display: flex; flex-wrap: wrap; gap: 1.3rem; font-size: .9rem; color: var(--muted); margin-bottom: 1.6rem; }
.dist-legend b { color: var(--text); font-weight: 600; margin-left: .35rem; }
.dist-legend i { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: .45rem; }

/* annotated article */
.paper {
    background: var(--surface); border: 1px solid var(--line); border-radius: 16px;
    padding: 1.9rem 2rem; font-family: var(--serif); font-size: 1.32rem; line-height: 2.15; color: #D3DAEA;
}
.hint { color: var(--muted); font-size: .84rem; margin-top: .7rem; }
.ent {
    --c: #fff; --soft: transparent; --i: 0;
    color: var(--text); font-weight: 500; padding: 1px 4px 2px; border-radius: 4px;
    border-bottom: 2px solid var(--c);
    background-image: linear-gradient(var(--soft), var(--soft));
    background-repeat: no-repeat; background-size: 0% 100%;
    animation: sweep .7s cubic-bezier(.2,.7,.2,1) forwards;
    animation-delay: calc(var(--i) * 120ms + 150ms);
}
.ent sup { font-family: var(--sans); font-size: .62rem; font-weight: 600; margin-left: 4px; color: var(--c); letter-spacing: .02em; }
@keyframes sweep { to { background-size: 100% 100%; } }
@media (prefers-reduced-motion: reduce) { .ent { animation: none; background-size: 100% 100%; } }

/* index */
.idx { border-left: 1px solid var(--line); padding-left: 1.4rem; }
.idx-h { display: flex; align-items: center; gap: .55rem; font-size: .95rem; font-weight: 600; color: var(--text); margin: 0 0 .4rem; }
.idx-h i { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.idx-h span { color: var(--muted); font-weight: 500; margin-left: auto; }
.idx-list { margin: 0 0 1.5rem; }
.idx-item { padding: .55rem 0; border-top: 1px solid var(--line); }
.idx-row { display: flex; justify-content: space-between; gap: .8rem; align-items: baseline; }
.idx-name { color: var(--text); font-weight: 500; word-break: break-word; }
.idx-meta { color: var(--muted); font-size: .8rem; white-space: nowrap; }
.mini { height: 3px; background: #1B2542; border-radius: 99px; margin-top: .5rem; overflow: hidden; }
.mini div { height: 100%; border-radius: 99px; }

.foot { margin-top: 3.5rem; padding-top: 1.2rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .85rem; display: flex; justify-content: space-between; flex-wrap: wrap; gap: .5rem; }
.foot a { color: var(--muted); text-underline-offset: 3px; }

@media (max-width: 760px) {
    .paper { padding: 1.2rem; font-size: 1.15rem; }
    .idx { border-left: 0; padding-left: 0; margin-top: 1.5rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


def render(markup: str):
    """Render HTML without Streamlit treating indented lines as a code block."""
    flat = " ".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


def notice(message: str):
    render(f'<div class="notice">{html.escape(message)}</div>')


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
                f'<div class="idx-item"><div class="idx-row">'
                f'<span class="idx-name">{html.escape(g["text"])}</span>'
                f'<span class="idx-meta">{extra}{g["score"]:.0%}</span></div>'
                f'<div class="mini"><div style="width:{g["score"]*100:.1f}%;background:{accent}"></div></div></div>'
            )
        parts.append(
            f'<div class="idx-h"><i style="background:{accent}"></i>{plural}<span>{len(items)}</span></div>'
            f'<div class="idx-list">{"".join(rows)}</div>'
        )
    return '<div class="idx">' + "".join(parts) + "</div>"


def set_sample(name):
    st.session_state["article_text"] = SAMPLES[name]
    st.session_state["result"] = None


def clear_all():
    st.session_state["article_text"] = ""
    st.session_state["result"] = None


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
  <div class="headline" role="heading" aria-level="1">See who and what a story is about.</div>
  <div class="lede">Paste a news article. NewsNER-AI marks every person, organization and place it
  names, and shows how sure it is about each one.</div>
</div>
"""
)


# ---------------------------------------------------------
# Input
# ---------------------------------------------------------

render('<div class="try">Try an example</div>')
chip_cols = st.columns([1.3, 1, 0.8, 4])
for col, name in zip(chip_cols, SAMPLES):
    with col:
        st.button(name, key=f"ex_{name}", on_click=set_sample, args=(name,), use_container_width=True)

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
        notice("Paste an article first, then select Find entities.")
    else:
        with st.spinner("Reading the article..."):
            entities, ms = run_ner(text)
        st.session_state["result"] = {"text": text, "entities": entities, "ms": ms}


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

result = st.session_state["result"]

if result:
    text = result["text"]
    entities = [e for e in result["entities"] if e["score"] >= MIN_CONFIDENCE]

    if not entities:
        notice("No named entities were found in this article. Try a longer or more specific text.")
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
            render('<div class="hint">Hover a highlight to see its type and confidence.</div>')
        with right:
            render(build_index(grouped))

            payload = [
                {"text": e["text"], "type": e["label"], "confidence": round(e["score"], 4),
                 "start": e["start"], "end": e["end"]}
                for e in entities
            ]
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=list(payload[0]))
            writer.writeheader()
            writer.writerows(payload)

            d1, d2 = st.columns(2)
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
