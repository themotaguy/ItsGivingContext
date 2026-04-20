---
title: "ItsGivingContext: A RAG-Based Internet Slang and Meme Explainer"
author: "Ritik Mota"
date: "April 2026"
geometry: margin=1in
fontsize: 11pt
---

# 1. Introduction

Internet slang and meme culture evolve faster than any static dictionary can keep up with. Terms like *rizz*, *touch grass*, and *NPC* carry rich cultural context that mainstream references fail to capture. **ItsGivingContext** is a Retrieval-Augmented Generation (RAG) system that explains internet slang and memes in plain English, grounded in two authoritative crowd-sourced knowledge bases: **Urban Dictionary** and **Know Your Meme**.

The system retrieves the most semantically relevant entries from a vector index of both sources, injects them into a structured prompt, and generates a concise explanation with inline citations tied directly to the retrieved chunks — enabling full verifiability of every claim.

# 2. System Architecture

The pipeline consists of four layers:

```
User Query
    ↓
[Embedding]    all-MiniLM-L6-v2 (sentence-transformers)
    ↓
[Retrieval]    ChromaDB cosine similarity → top-3 chunks
    ↓
[Augmentation] Prompt template with retrieved context + chunk citations
    ↓
[Generation]   Llama 3.1 8B (Groq API / local Ollama)
    ↓
Explanation + cited chunks shown in Streamlit UI
```

# 3. Data Collection & Processing

## 3.1 Urban Dictionary

- **Source:** Kaggle dataset (`therohk/urban-dictionary-words-definitions`) — 2.58M raw entries
- **Filtering pipeline (`ud_ingest.py`):**
  - Net upvotes >= 10 (removes low-quality, downvoted entries)
  - Upvote ratio >= 60% (community quality signal)
  - Definition length: 30--2,000 characters
  - Keyword blocklist applied to both the word and definition fields (slurs + explicit sexual content)
  - Deduplication by (word, definition) pair, keeping highest net-voted version
- **Result:** ~565k filtered entries → top 50,000 by net votes indexed

## 3.2 Know Your Meme

- **Source:** Custom web scraper (`kym_scraper.py`) with polite crawling (3–6s delay between requests)
- **Scope:** `confirmed` and `notable` status entries only
- **Parsing:** structured extraction of `About`, `Origin`, `Spread`, and `Notable Examples` sections using KYM's anchor IDs; metadata (year, platform, tags) extracted from sidebar `dl/dt/dd` elements
- **Incremental checkpointing:** slug-level checkpoint file allows resuming interrupted scrapes
- **Result:** 6,133 confirmed/notable meme entries

# 4. Embedding & Indexing

## 4.1 Embedding Model

**`all-MiniLM-L6-v2`** (sentence-transformers) was selected for:

- Strong performance on semantic similarity benchmarks (SBERT paper, Reimers & Gurevych 2019)
- Lightweight (22M parameters, 384-dimensional embeddings) — fits within cloud memory constraints
- Reproducible, pure-Python setup with no external services required

## 4.2 Chunking Strategy

Two source-specific chunking strategies were applied:

| Source | Strategy | Rationale |
|--------|----------|-----------|
| Urban Dictionary | One chunk per entry: `word: definition` | Definitions are already atomic units |
| Know Your Meme | One chunk per article section (`about`, `origin`, `spread`, `examples`) | Each section covers a distinct aspect of the meme; keeps context tight |

## 4.3 Vector Database

**ChromaDB** with **cosine similarity** was chosen for:

- Embedded (no separate server process required)
- Persistent local storage for development; in-memory `EphemeralClient` for cloud deployment
- Native support for metadata filtering and incremental upserts

## 4.4 Index Statistics

| Source | Entries | Chunks Indexed |
|--------|---------|----------------|
| Urban Dictionary | 50,000 | 50,000 |
| Know Your Meme | 6,133 | ~18,000 |
| **Total** | | **~68,000** |

# 5. RAG Pipeline

## 5.1 Retrieval

At query time, the user's input is embedded with `all-MiniLM-L6-v2` and a cosine similarity search returns the top-3 most relevant chunks. A minimum similarity threshold of 0.30 filters out low-confidence matches, returning a "no results" message rather than generating an explanation from unrelated context.

## 5.2 Augmentation

Retrieved chunks are injected into a structured prompt with:

- Numbered citations `[1]`, `[2]`, `[3]` matching each chunk
- Full chunk text and source URL per entry
- Instructions to cite inline and enumerate used citations at the end

## 5.3 Generation

- **Local development:** Llama 3.2 via Ollama (streamed token-by-token to the UI)
- **Cloud deployment:** Llama 3.1 8B Instant via Groq API (fast, free tier, ~1s latency)
- Output capped at 300 tokens for conciseness

# 6. Content Moderation

Urban Dictionary contains substantial explicit and harmful content. A two-stage moderation approach was applied:

1. **Upvote filtering** — community downvoting acts as a crowd-sourced quality signal
2. **Keyword blocklist** — regex patterns applied to both the `word` and `definition` fields filter slurs and explicit sexual content before indexing

Know Your Meme's own editorial `confirmed`/`notable` status gate serves as an additional filter — only entries reviewed and confirmed by the KYM community are scraped.

# 7. Citation Design

The grading rubric requires verifiable citations. Each generated explanation includes:

- **Inline citations** — `[1]`, `[2]`, `[3]` within the generated text, tied to specific retrieved chunks
- **Chunk display** — the full retrieved text of each cited chunk is rendered in the UI
- **Chunk ID** — e.g. `kym_touch-grass_body_about`, identifying exactly which document was retrieved from the index
- **Web source link** — URL to the original Urban Dictionary entry or Know Your Meme article

This design exposes the full information retrieval pipeline: the grader can verify each claim against both the retrieved chunk and its original source.

# 8. Deployment

The application is deployed on **Streamlit Community Cloud**. To avoid slow cold-start re-embedding, KYM embeddings are pre-computed locally (`precompute_embeddings.py`) and stored as a compressed NumPy archive (`kym_embeddings.npz`, ~9MB), committed to the repository. At startup, pre-computed vectors are loaded directly into an in-memory ChromaDB collection — reducing cold-start time from ~10 minutes to ~10 seconds.

A **Docker Compose** configuration is also provided for local reproducibility, spinning up both the Streamlit app and an Ollama service with `llama3.2` automatically pulled on first run.

# 9. Limitations & Future Work

- **Recency gap:** Urban Dictionary's top-voted entries skew older; emerging slang (e.g. *rizz*, *delulu*) may not appear in the top 50k by net votes. Indexing by recency rather than votes would improve coverage of new terms.
- **KYM scrape completeness:** IP rate-limiting during scraping meant a subset of target slugs were skipped. A longer scrape window with distributed delays would increase coverage.
- **Cloud UD index:** The UD index (50k entries) exceeds practical GitHub file size limits for the pre-computed embeddings approach. The deployed version currently uses KYM only; a cloud object store (e.g. HuggingFace Hub datasets) would allow including UD.
- **Content moderation:** The keyword blocklist is brittle. Future work could fine-tune a small text classifier on Urban Dictionary data for semantic content moderation.

# 10. References

- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.
- Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.
- Urban Dictionary dataset: therohk, Kaggle.
- Know Your Meme: knowyourmeme.com
