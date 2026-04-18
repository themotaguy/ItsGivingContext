"""
Embedding + ChromaDB indexing pipeline for ItsGivingContext RAG.

Supports incremental indexing — run once for Urban Dictionary now,
run again after KYM scrape completes to add meme entries.

Usage:
    python3 build_index.py --source ud        # index Urban Dictionary
    python3 build_index.py --source kym       # index Know Your Meme (after scrape)
    python3 build_index.py --source both      # index both

Requirements:
    pip install sentence-transformers chromadb pandas tqdm
"""

import argparse
import textwrap
import pandas as pd
import chromadb
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR      = Path("data/chroma_db")
COLLECTION_NAME = "slang_memes"
EMBED_MODEL     = "all-MiniLM-L6-v2"
BATCH_SIZE      = 64    # small batches to avoid memory crashes
UD_MAX_ROWS     = 50_000  # top entries by net_votes — enough for RAG, fits in RAM

UD_PATH  = Path("data/urban_dictionary_filtered.csv")
KYM_PATH = Path("data/know_your_meme.csv")

# ── ChromaDB client ───────────────────────────────────────────────────────────
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection

# ── Batch indexing (shared by UD and KYM) ────────────────────────────────────
def load_existing_ids(collection) -> set:
    """Load all existing IDs from the collection in one query."""
    result = collection.get(include=[])  # fetch IDs only, no documents/embeddings
    return set(result["ids"])

def index_batch(ids, texts, metadatas, model, collection, existing_ids: set):
    """Embed and add a batch, skipping IDs already in the collection."""
    new = [(i, t, m) for i, t, m in zip(ids, texts, metadatas) if i not in existing_ids]
    if not new:
        return 0
    n_ids, n_texts, n_metas = zip(*new)
    embeddings = model.encode(list(n_texts), show_progress_bar=False).tolist()
    collection.add(
        ids=list(n_ids),
        embeddings=embeddings,
        documents=list(n_texts),
        metadatas=list(n_metas),
    )
    existing_ids.update(n_ids)
    return len(new)

# ── Urban Dictionary ──────────────────────────────────────────────────────────
def index_ud(model, collection, existing_ids: set):
    print(f"[Urban Dictionary] Loading top {UD_MAX_ROWS:,} entries by net votes...")
    df = pd.read_csv(UD_PATH, dtype={"word_id": str}, on_bad_lines="skip", engine="python")
    df = df.nlargest(UD_MAX_ROWS, "net_votes").reset_index(drop=True)
    print(f"  {len(df):,} entries selected.")

    total_added = 0
    ids, texts, metas = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  UD entries"):
        word = str(row["word"]).strip()
        defn = str(row["definition"]).strip()
        ids.append(f"ud_{row['word_id']}")
        texts.append(f"{word}: {defn}")
        metas.append({
            "source":     "urban_dictionary",
            "word":       word,
            "definition": defn,
            "up_votes":   int(row.get("up_votes", 0)),
            "down_votes": int(row.get("down_votes", 0)),
            "net_votes":  int(row.get("net_votes", 0)),
            "url":        f"https://www.urbandictionary.com/define.php?term={word.replace(' ', '+')}",
        })
        if len(ids) >= BATCH_SIZE:
            total_added += index_batch(ids, texts, metas, model, collection, existing_ids)
            ids, texts, metas = [], [], []

    if ids:
        total_added += index_batch(ids, texts, metas, model, collection, existing_ids)

    print(f"  Added {total_added:,} new UD entries. Collection size: {collection.count():,}\n")

# ── Know Your Meme ────────────────────────────────────────────────────────────
def index_kym(model, collection, existing_ids: set):
    if not KYM_PATH.exists():
        print(f"[KYM] {KYM_PATH} not found — run kym_scraper.py first.")
        return

    print(f"[Know Your Meme] Loading {KYM_PATH} ...")
    df = pd.read_csv(KYM_PATH)
    print(f"  {len(df):,} entries loaded.")

    sections = ["body_about", "body_origin", "body_spread", "body_examples"]
    ids, texts, metas = [], [], []
    total_added = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  KYM entries"):
        title = str(row["title"]).strip()
        slug  = str(row["slug"]).strip()
        url   = str(row["url"]).strip()
        for sec in sections:
            text = str(row.get(sec, "")).strip()
            if len(text) < 50:
                continue
            text = textwrap.shorten(text, width=1500, placeholder="...")
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
            if len(ids) >= BATCH_SIZE:
                total_added += index_batch(ids, texts, metas, model, collection, existing_ids)
                ids, texts, metas = [], [], []

    if ids:
        total_added += index_batch(ids, texts, metas, model, collection, existing_ids)

    print(f"  Added {total_added:,} new KYM entries. Collection size: {collection.count():,}\n")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["ud", "kym", "both"], default="ud")
    args = parser.parse_args()

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    collection   = get_collection()
    existing_ids = load_existing_ids(collection)
    print(f"Collection '{COLLECTION_NAME}' — current size: {len(existing_ids):,}\n")

    if args.source in ("ud", "both"):
        index_ud(model, collection, existing_ids)

    if args.source in ("kym", "both"):
        index_kym(model, collection, existing_ids)

    print("Indexing complete.")

if __name__ == "__main__":
    main()
