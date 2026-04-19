"""
Pre-compute KYM embeddings and save to disk for fast cloud startup.

Run once locally:
    python3 precompute_embeddings.py

Output: data/kym_embeddings.npz
    - embeddings: float32 array (N, 384)
    - texts: array of document strings
    - metadatas: array of JSON-encoded metadata dicts
    - ids: array of document IDs
"""

import json
import textwrap
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pathlib import Path

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

CSV_PATH   = Path("data/know_your_meme.csv")
OUT_PATH   = Path("data/kym_embeddings.npz")
EMBED_MODEL = "all-MiniLM-L6-v2"
SECTIONS    = ["body_about", "body_origin", "body_spread", "body_examples"]
BATCH_SIZE  = 128

print(f"Loading {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH)
print(f"  {len(df):,} entries")

print(f"Loading embedding model: {EMBED_MODEL}")
model = SentenceTransformer(EMBED_MODEL)

ids, texts, metas = [], [], []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Building chunks"):
    title = str(row["title"]).strip()
    slug  = str(row["slug"]).strip()
    url   = str(row["url"]).strip()
    for sec in SECTIONS:
        text = str(row.get(sec, "")).strip()
        if len(text) < 50:
            continue
        text          = textwrap.shorten(text, width=1500, placeholder="...")
        section_label = sec.replace("body_", "").replace("_", " ")
        ids.append(f"kym_{slug}_{sec}")
        texts.append(f"{title} ({section_label}): {text}")
        metas.append(json.dumps({
            "source":  "know_your_meme",
            "title":   title,
            "slug":    slug,
            "section": section_label,
            "tags":    str(row.get("tags", "")),
            "year":    str(row.get("year", "")),
            "url":     url,
        }))

print(f"\nEmbedding {len(texts):,} chunks ...")
embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)

np.savez_compressed(
    OUT_PATH,
    embeddings=embeddings.astype(np.float32),
    texts=np.array(texts),
    metadatas=np.array(metas),
    ids=np.array(ids),
)
print(f"\nSaved → {OUT_PATH}  ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")
