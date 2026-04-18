"""
RAG query engine for ItsGivingContext.

Given a user query (slang term or meme), retrieves the most relevant
entries from ChromaDB and generates an explanation with citations.

Requirements:
    pip install sentence-transformers chromadb requests
    brew install ollama && ollama pull llama3.2
"""

import os
import requests
import chromadb

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR      = Path("data/chroma_db")
COLLECTION_NAME = "slang_memes"
EMBED_MODEL     = "all-MiniLM-L6-v2"
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL      = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
TOP_K            = 3    # number of results to retrieve
MIN_SIMILARITY   = 0.30  # drop results below this threshold

# ── Clients (lazy-loaded singletons) ─────────────────────────────────────────
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

# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """Embed query and return top-k matching documents with metadata."""
    embedder   = get_embedder()
    collection = get_collection()

    query_vec = embedder.encode([query]).tolist()
    results   = collection.query(
        query_embeddings=query_vec,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = round(1 - dist, 4)
        if similarity >= MIN_SIMILARITY:
            hits.append({
                "text":       doc,
                "metadata":   meta,
                "similarity": similarity,
            })
    return hits

# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(query: str, hits: list[dict]) -> str:
    context_blocks = []
    for i, hit in enumerate(hits, 1):
        meta   = hit["metadata"]
        source = meta.get("source", "")
        url    = meta.get("url", "")

        if source == "urban_dictionary":
            word = meta.get('word', '')
            header = f'[{i}] Urban Dictionary -- "{word}"'
        else:
            title   = meta.get("title", "")
            section = meta.get("section", "")
            header  = f'[{i}] Know Your Meme -- "{title}" ({section})'

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

# ── Main query function ───────────────────────────────────────────────────────
def explain(query: str) -> dict:
    """
    Full RAG pipeline: retrieve → prompt → generate.
    Returns a dict with keys: query, explanation, sources, hits.
    """
    hits = retrieve(query)
    if not hits:
        return {
            "query":       query,
            "explanation": f'Sorry, no relevant results found for "{query}". It may be too new or too niche for our dataset.',
            "sources":     [],
            "hits":        [],
        }
    prompt = build_prompt(query, hits)

    response = requests.post(
        OLLAMA_URL,
        json={
            "model":   OLLAMA_MODEL,
            "prompt":  prompt,
            "stream":  False,
            "options": {"num_predict": 300},  # cap output length
        },
        timeout=120,
    )
    response.raise_for_status()
    explanation = response.json()["response"]

    # Extract source URLs from hits for structured return
    sources = []
    for hit in hits:
        url = hit["metadata"].get("url", "")
        if url and url not in sources:
            sources.append(url)

    return {
        "query":       query,
        "explanation": explanation,
        "sources":     sources,
        "hits":        hits,
    }


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "no cap"
    result = explain(query)
    print(f"\n=== {result['query']} ===\n")
    print(result["explanation"])
