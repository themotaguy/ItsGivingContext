"""
ItsGivingContext — Streamlit frontend.

Run with:
    streamlit run app.py
"""

import requests
import streamlit as st
from rag import retrieve, build_prompt, OLLAMA_URL, OLLAMA_MODEL, MIN_SIMILARITY

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ItsGivingContext",
    page_icon="🔍",
    layout="centered",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 ItsGivingContext")
st.caption("Explain internet slang and memes — powered by Urban Dictionary & Know Your Meme")

# ── Input ─────────────────────────────────────────────────────────────────────
with st.form("search_form"):
    query = st.text_input(
        label="Enter a slang term or meme",
        placeholder="e.g. no cap, rizz, NPC, touch grass...",
        label_visibility="collapsed",
    )
    search = st.form_submit_button("Explain it", type="primary", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
if search and query.strip():
    hits = retrieve(query.strip())

    st.markdown("---")
    st.subheader(f"What is \"{query}\"?")

    if not hits:
        st.warning(f'No relevant results found for "{query}". It may be too new or niche for our dataset.')
        st.stop()

    prompt = build_prompt(query.strip(), hits)

    # Stream the response token by token
    def stream_ollama():
        with requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True,
                  "options": {"num_predict": 300}},
            stream=True, timeout=120,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    import json
                    chunk = json.loads(line)
                    yield chunk.get("response", "")
                    if chunk.get("done"):
                        break

    try:
        explanation = st.write_stream(stream_ollama())
    except Exception as e:
        st.error(f"Ollama error: {e}. Make sure `ollama serve` is running.")

    # Retrieved sources — deduplicate by URL
    st.markdown("---")
    st.subheader("Sources")
    seen_urls = set()
    citation  = 1
    for hit in result["hits"]:
        meta = hit["metadata"]
        url  = meta.get("url", "#")
        if url in seen_urls:
            continue
        seen_urls.add(url)

        source = meta.get("source", "")
        sim    = hit["similarity"]

        if source == "urban_dictionary":
            label = f"**[{citation}] Urban Dictionary** — *{meta.get('word', '')}*"
        else:
            label = f"**[{citation}] Know Your Meme** — *{meta.get('title', '')}* ({meta.get('section', '')})"

        with st.expander(f"{label}  `similarity: {sim}`"):
            st.markdown(f"[View source]({url})")
            st.caption(hit["text"][:500] + ("..." if len(hit["text"]) > 500 else ""))
        citation += 1

elif search and not query.strip():
    st.warning("Please enter a term to look up.")
