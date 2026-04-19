"""
ItsGivingContext — Streamlit frontend.

Run locally:
    streamlit run app.py

Cloud deployment:
    Set HF_API_TOKEN in Streamlit secrets.
    The index is built at startup from data/know_your_meme.csv.
"""

import json
import os
import requests
import streamlit as st

# Inject Streamlit secrets into env vars before importing rag
if "HF_API_TOKEN" in st.secrets:
    os.environ["HF_API_TOKEN"] = st.secrets["HF_API_TOKEN"]

from rag import (
    retrieve, build_prompt, build_collection_from_kym,
    get_persistent_collection, stream_ollama, generate_hf,
    OLLAMA_URL, OLLAMA_MODEL, HF_API_TOKEN,
)

st.set_page_config(
    page_title="ItsGivingContext",
    page_icon="🔍",
    layout="centered",
)

# ── Startup: build or load the index ─────────────────────────────────────────
@st.cache_resource(show_spinner="Building index from Know Your Meme data...")
def load_collection():
    """
    On cloud: build an in-memory index from the KYM CSV.
    Locally: load the pre-built ChromaDB from disk.
    """
    if os.path.exists("data/chroma_db"):
        return get_persistent_collection()
    return build_collection_from_kym("data/know_your_meme.csv")

collection = load_collection()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 ItsGivingContext")
st.caption("Explain internet slang and memes — powered by Know Your Meme")

# ── Input ─────────────────────────────────────────────────────────────────────
with st.form("search_form"):
    query  = st.text_input(
        label="Enter a slang term or meme",
        placeholder="e.g. no cap, rizz, NPC, touch grass...",
        label_visibility="collapsed",
    )
    search = st.form_submit_button("Explain it", type="primary", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
if search and query.strip():
    q    = query.strip()
    hits = retrieve(q, collection=collection)

    st.markdown("---")
    st.subheader(f"What is \"{q}\"?")

    if not hits:
        st.warning(f'No relevant results found for "{q}". It may be too new or niche for our dataset.')
        st.stop()

    prompt = build_prompt(q, hits)

    if os.environ.get("HF_API_TOKEN", HF_API_TOKEN):
        # Cloud: HuggingFace Inference API (non-streaming)
        with st.spinner("Generating explanation..."):
            try:
                explanation = generate_hf(prompt)
                st.markdown(explanation)
            except Exception as e:
                st.error(f"HuggingFace API error: {e}")
                st.stop()
    else:
        # Local: Ollama streaming
        try:
            st.write_stream(stream_ollama(prompt))
        except Exception as e:
            st.error(f"Ollama error: {e}. Make sure `ollama serve` is running.")
            st.stop()

    # Citations — show retrieved chunks with full text and web links
    st.markdown("---")
    st.subheader("Retrieved Chunks (Citations)")
    seen_urls = set()
    citation  = 1
    for hit in hits:
        meta    = hit["metadata"]
        url     = meta.get("url", "#")
        source  = meta.get("source", "")
        sim     = hit["similarity"]
        chunk_id = f"kym_{meta.get('slug','')}_{meta.get('section','').replace(' ','_')}" \
                   if source == "know_your_meme" \
                   else f"ud_{meta.get('word','').replace(' ','_')}"

        if source == "urban_dictionary":
            label = f"[{citation}] Urban Dictionary — *{meta.get('word', '')}*"
        else:
            label = f"[{citation}] Know Your Meme — *{meta.get('title', '')}* ({meta.get('section', '')})"

        with st.expander(f"{label}  `sim: {sim}`", expanded=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"Chunk ID: `{chunk_id}`")
            with col2:
                if url != "#":
                    st.markdown(f"[View web source ↗]({url})")
            st.markdown(hit["text"])

        if url not in seen_urls:
            seen_urls.add(url)
        citation += 1

elif search and not query.strip():
    st.warning("Please enter a term to look up.")
