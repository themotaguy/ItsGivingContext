# ItsGivingContext

A Retrieval-Augmented Generation (RAG) system that explains internet slang and memes using Urban Dictionary and Know Your Meme as knowledge sources.

## How it works

1. **Data ingestion** — Urban Dictionary definitions are filtered for quality (upvote ratio, length, content moderation). Know Your Meme articles are scraped and parsed by section (About, Origin, Spread, Examples).
2. **Indexing** — Entries are embedded using `all-MiniLM-L6-v2` (sentence-transformers) and stored in a ChromaDB vector database.
3. **Retrieval** — User queries are embedded and matched against the index using cosine similarity.
4. **Generation** — Retrieved entries are injected into a prompt and passed to a local LLM (Llama 3.2 via Ollama) to generate a plain-English explanation with citations.
5. **Frontend** — A Streamlit app provides a search interface with clickable source links.

## Setup

### Requirements
- Python 3.11
- [Ollama](https://ollama.com) with `llama3.2` model

### Installation

```bash
# Create and activate virtual environment (Python 3.11 required)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install and start Ollama
brew install ollama
ollama pull llama3.2
ollama serve   # run in a separate terminal
```

### Data

The pre-built ChromaDB index is not included in the repo. To rebuild it:

```bash
# 1. Filter Urban Dictionary
#    Download urbandict-word-defs.csv from Kaggle first:
#    https://www.kaggle.com/datasets/therohk/urban-dictionary-words-definitions
python3 ud_ingest.py

# 2. Scrape Know Your Meme
python3 kym_scraper.py

# 3. Build the vector index
python3 build_index.py --source both
```

## Running the app

```bash
source .venv/bin/activate
streamlit run app.py
```

## Project structure

```
├── app.py              # Streamlit frontend
├── rag.py              # RAG query engine (retrieval + generation)
├── build_index.py      # Embedding + ChromaDB indexing pipeline
├── ud_ingest.py        # Urban Dictionary ingestion + filtering
├── kym_scraper.py      # Know Your Meme scraper
├── requirements.txt
└── data/
    ├── urbandict-word-defs.csv         # Raw UD data (not tracked)
    ├── urban_dictionary_filtered.csv   # Filtered UD data (not tracked)
    ├── know_your_meme.csv              # Scraped KYM data (not tracked)
    └── chroma_db/                      # Vector index (not tracked)
```

## Dataset stats

| Source | Raw entries | After filtering | Indexed |
|--------|-------------|-----------------|---------|
| Urban Dictionary | ~2.58M | ~565k | 50k (top by net votes) |
| Know Your Meme | ~6.1k confirmed/notable | — | ~18k chunks |
