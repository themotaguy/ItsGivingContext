"""
RAG query engine for ItsGivingContext.

Supports two LLM backends:
  - HuggingFace Inference API  (cloud deployment — set HF_API_TOKEN)
  - Ollama                     (local — run `ollama serve`)

Backend is selected automatically: HF if HF_API_TOKEN is set, else Ollama.

Requirements:
    pip install sentence-transformers chromadb requests
"""

import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import textwrap
import requests
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR      = Path("data/chroma_db")
COLLECTION_NAME = "slang_memes"
EMBED_MODEL     = "all-MiniLM-L6-v2"
TOP_K           = 3
MIN_SIMILARITY  = 0.30

# Ollama (local)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate"

# HuggingFace Inference API (cloud)
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
HF_MODEL     = "mistralai/Mistral-7B-Instruct-v0.2"
HF_API_URL   = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

_embedder = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def get_persistent_collection():
    """Load the pre-built ChromaDB from disk (local use)."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)

def build_collection_from_kym(csv_path: str = "data/know_your_meme.csv"):
    """
    Build an in-memory ChromaDB collection from the KYM CSV at startup.
    Used for cloud deployment where the pre-built index isn't available.
    """
    import pandas as pd
    from tqdm import tqdm

    client     = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    model    = get_embedder()
    df       = pd.read_csv(csv_path)
    sections = ["body_about", "body_origin", "body_spread", "body_examples"]
    BATCH    = 64

    ids, texts, metas = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing KYM"):
        title = str(row["title"]).strip()
        slug  = str(row["slug"]).strip()
        url   = str(row["url"]).strip()
        for sec in sections:
            text = str(row.get(sec, "")).strip()
            if len(text) < 50:
                continue
            text          = textwrap.shorten(text, width=1500, placeholder="...")
            section_label = sec.replace("body_", "").replace("_", " ")
            ids.append(f"kym_{slug}_{sec}")
            texts.append(f"{title} ({section_label}): {text}")
            metas.append({
                "source":  "know_your_meme",
                "title":   title,
                "slug":    slug,
                "section": section_label,
                "tags":    str(row.get("tags", "")),
                "year":    str(row.get("year", "")),
                "url":     url,
            })
            if len(ids) >= BATCH:
                embs = model.encode(texts, show_progress_bar=False).tolist()
                collection.add(ids=ids, embeddings=embs, documents=texts, metadatas=metas)
                ids, texts, metas = [], [], []

    if ids:
        embs = model.encode(texts, show_progress_bar=False).tolist()
        collection.add(ids=ids, embeddings=embs, documents=texts, metadatas=metas)

    return collection

# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str, collection=None, top_k: int = TOP_K) -> list[dict]:
    """Embed query and return top-k matching documents."""
    if collection is None:
        collection = get_persistent_collection()
    query_vec = get_embedder().encode([query]).tolist()
    results   = collection.query(
        query_embeddings=query_vec,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": doc, "metadata": meta, "similarity": round(1 - dist, 4)}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
        if round(1 - dist, 4) >= MIN_SIMILARITY
    ]

# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(query: str, hits: list[dict]) -> str:
    context_blocks = []
    for i, hit in enumerate(hits, 1):
        meta   = hit["metadata"]
        source = meta.get("source", "")
        url    = meta.get("url", "")
        if source == "urban_dictionary":
            header = f'[{i}] Urban Dictionary -- "{meta.get("word", "")}"'
        else:
            header = f'[{i}] Know Your Meme -- "{meta.get("title", "")}" ({meta.get("section", "")})'
        context_blocks.append(f"{header}\nURL: {url}\n{hit['text']}")

    context = "\n\n---\n\n".join(context_blocks)
    return f"""You are a helpful cultural explainer that specializes in internet slang, memes, and Gen Z language.

A user wants to understand: "{query}"

Here are the most relevant reference entries from Urban Dictionary and Know Your Meme, numbered for citation:

{context}

Using ONLY the information in these entries, write a clear, friendly explanation of "{query}".
Your explanation should:
- Define what it means in plain English
- Mention where it came from or how it spread (if the sources say)
- Give an example of how it's used in a sentence
- Be 3-5 sentences, conversational in tone
- Cite the entry numbers inline wherever you use information, like this: "no cap means no lie [1]."

End your response with a line listing all cited entry numbers, like:
Citations: [1] [2]

If the entries don't contain enough information to explain the term, say so honestly."""

# ── Generation ────────────────────────────────────────────────────────────────
def generate_hf(prompt: str) -> str:
    """Generate via HuggingFace Inference API (non-streaming)."""
    response = requests.post(
        HF_API_URL,
        headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
        json={
            "inputs":      f"[INST] {prompt} [/INST]",
            "parameters":  {"max_new_tokens": 300, "return_full_text": False},
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()[0]["generated_text"]

def stream_ollama(prompt: str):
    """Stream generation from local Ollama (generator)."""
    with requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True,
              "options": {"num_predict": 300}},
        stream=True, timeout=120,
    ) as r:
        import json
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                chunk = json.loads(line)
                yield chunk.get("response", "")
                if chunk.get("done"):
                    break

def explain(query: str) -> dict:
    """Full RAG pipeline for CLI use."""
    hits = retrieve(query)
    if not hits:
        return {"query": query,
                "explanation": f'No relevant results found for "{query}".',
                "sources": [], "hits": []}
    prompt      = build_prompt(query, hits)
    explanation = generate_hf(prompt) if HF_API_TOKEN else "".join(stream_ollama(prompt))
    sources     = list(dict.fromkeys(
        h["metadata"]["url"] for h in hits if h["metadata"].get("url")
    ))
    return {"query": query, "explanation": explanation, "sources": sources, "hits": hits}


if __name__ == "__main__":
    import sys
    query  = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "no cap"
    result = explain(query)
    print(f"\n=== {result['query']} ===\n")
    print(result["explanation"])
