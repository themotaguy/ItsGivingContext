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

# Groq Inference API (cloud — free tier)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.1-8b-instant"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

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

def build_collection_from_kym(embeddings_path: str = "data/kym_embeddings.npz"):
    """
    Load pre-computed KYM embeddings into an in-memory ChromaDB collection.
    Fast startup — no re-embedding needed.
    Run precompute_embeddings.py once locally to generate the .npz file.
    """
    import json
    import numpy as np

    data       = np.load(embeddings_path, allow_pickle=True)
    ids        = data["ids"].tolist()
    texts      = data["texts"].tolist()
    embeddings = data["embeddings"].tolist()
    metadatas  = [json.loads(m) for m in data["metadatas"].tolist()]

    client     = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    BATCH = 512
    for i in range(0, len(ids), BATCH):
        collection.add(
            ids=ids[i:i+BATCH],
            embeddings=embeddings[i:i+BATCH],
            documents=texts[i:i+BATCH],
            metadatas=metadatas[i:i+BATCH],
        )

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
def generate_groq(prompt: str) -> str:
    """Generate via Groq API (free tier, fast)."""
    token = os.environ.get("GROQ_API_KEY", GROQ_API_KEY)
    response = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "model":    GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

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
    explanation = generate_groq(prompt) if os.environ.get("GROQ_API_KEY", GROQ_API_KEY) else "".join(stream_ollama(prompt))
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
