"""
RAG query engine for ItsGivingContext.

Given a user query (slang term or meme), retrieves the most relevant
entries from ChromaDB and generates an explanation with citations.

Requirements:
    pip install sentence-transformers chromadb requests
    brew install ollama && ollama pull llama3.2
"""

import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import requests
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR      = Path("data/chroma_db")
COLLECTION_NAME = "slang_memes"
EMBED_MODEL     = "all-MiniLM-L6-v2"
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL      = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
TOP_K           = 3
MIN_SIMILARITY  = 0.30

_embedder   = None
_collection = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Embed query and return top-k matching documents with metadata."""
    query_vec = get_embedder().encode([query]).tolist()
    results   = get_collection().query(
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

Here are the most relevant reference entries from Urban Dictionary and Know Your Meme:

{context}

Using ONLY the information in these entries, write a clear, friendly explanation of "{query}".
Your explanation should:
- Define what it means in plain English
- Mention where it came from or how it spread (if the sources say)
- Give an example of how it's used in a sentence
- Be 3-5 sentences, conversational in tone

At the end, list the sources you used as clickable citations in this format:
Sources: [1] <url1>  [2] <url2>  ...

If the entries don't contain enough information to explain the term, say so honestly."""

def explain(query: str) -> dict:
    """Full RAG pipeline for CLI use: retrieve → prompt → generate."""
    hits = retrieve(query)
    if not hits:
        return {
            "query":       query,
            "explanation": f'Sorry, no relevant results found for "{query}".',
            "sources":     [],
            "hits":        [],
        }
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": build_prompt(query, hits),
              "stream": False, "options": {"num_predict": 300}},
        timeout=120,
    )
    response.raise_for_status()
    explanation = response.json()["response"]
    sources = list(dict.fromkeys(
        h["metadata"]["url"] for h in hits if h["metadata"].get("url")
    ))
    return {"query": query, "explanation": explanation, "sources": sources, "hits": hits}


if __name__ == "__main__":
    import sys
    query  = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "no cap"
    result = explain(query)
    print(f"\n=== {result['query']} ===\n")
    print(result["explanation"])
