"""
Urban Dictionary ingestion + filtering pipeline.

Filters the raw CSV down to high-quality, safe-ish entries suitable for RAG.

Filters applied:
  1. Minimum net upvotes (up_votes - down_votes >= MIN_NET_VOTES)
  2. Minimum upvote ratio (up_votes / (up_votes + down_votes) >= MIN_RATIO)
  3. Minimum definition length (characters)
  4. Keyword blocklist on the definition text

Output: data/urban_dictionary_filtered.csv

Requirements:
    pip install pandas tqdm
"""

import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_PATH  = Path("data/urbandict-word-defs.csv")
OUTPUT_PATH = Path("data/urban_dictionary_filtered.csv")

MIN_NET_VOTES   = 10       # up - down must be at least this
MIN_RATIO       = 0.60     # upvote ratio threshold
MIN_DEF_LENGTH  = 30       # minimum characters in definition
MAX_DEF_LENGTH  = 2000     # strip runaway definitions

# Blocklist — slurs + explicit sexual content
BLOCKLIST = [
    # Slurs
    r"\bn[i!1]gg[ae]r\b",
    r"\bfagg?[o0]t\b",
    r"\bch[i!1]nk\b",
    r"\bsp[i!1]c\b",
    r"\bk[i!1]ke\b",
    r"\bretard\b",
    # Child safety
    r"\bcp\b",
    r"child porn",
    r"rape porn",
    # Explicit sexual content
    r"\bpenis\b",
    r"\bvagina\b",
    r"\bcock\b",
    r"\bpussy\b",
    r"\bdick\b",
    r"\bcum\b",
    r"\bjizz\b",
    r"\bsemen\b",
    r"\bboner\b",
    r"\berection\b",
    r"\bblow\s?job\b",
    r"\bhand\s?job\b",
    r"\bjerk\s?off\b",
    r"\bjack\s?off\b",
    r"\bmasturbat",
    r"\bsex\b",
    r"\bfuck(ing)?\b",
    r"\bfucked\b",
    r"\bfucker\b",
    r"\bshit\b",
    r"\basshole\b",
    r"\bbitch\b",
    r"\bwhore\b",
    r"\bslut\b",
    r"\bporn\b",
    r"\bhentai\b",
    r"\bgangbang\b",
    r"\banal\b",
    r"\bnaked\b",
    r"\bnude\b",
    r"\bboob\b",
    r"\bbreast\b",
    r"\bbutt\b",
    r"\bass\b",
    r"\bhorny\b",
    r"\baroused\b",
    r"\bintercourse\b",
    r"\bscrewing\b",
    r"\bballing\b",
    r"\bsucking\b",
    r"\bsuck(s)?\b",
]
BLOCKLIST_RE = re.compile("|".join(BLOCKLIST), re.IGNORECASE)

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading {INPUT_PATH} ...")
df = pd.read_csv(INPUT_PATH, dtype={"word_id": str}, on_bad_lines="skip", engine="python")
print(f"Raw rows: {len(df):,}")

# ── Clean ─────────────────────────────────────────────────────────────────────
# Drop rows missing key fields
df = df.dropna(subset=["word", "definition"])
df["word"]       = df["word"].astype(str).str.strip()
df["definition"] = df["definition"].astype(str).str.strip()

# Numeric columns — coerce bad values to 0
df["up_votes"]   = pd.to_numeric(df["up_votes"],   errors="coerce").fillna(0).astype(int)
df["down_votes"] = pd.to_numeric(df["down_votes"], errors="coerce").fillna(0).astype(int)

df["net_votes"]     = df["up_votes"] - df["down_votes"]
df["total_votes"]   = df["up_votes"] + df["down_votes"]
df["upvote_ratio"]  = df["up_votes"] / df["total_votes"].replace(0, 1)
df["def_length"]    = df["definition"].str.len()

print(f"After cleaning: {len(df):,}")

# ── Filter ────────────────────────────────────────────────────────────────────
mask = (
    (df["net_votes"]    >= MIN_NET_VOTES)  &
    (df["upvote_ratio"] >= MIN_RATIO)      &
    (df["def_length"]   >= MIN_DEF_LENGTH) &
    (df["def_length"]   <= MAX_DEF_LENGTH)
)
df = df[mask].copy()
print(f"After vote/length filter: {len(df):,}")

# Blocklist filter — check both word and definition
tqdm.pandas(desc="Blocklist filtering")
safe_mask = ~(
    df["definition"].progress_apply(lambda x: bool(BLOCKLIST_RE.search(x))) |
    df["word"].apply(lambda x: bool(BLOCKLIST_RE.search(x)))
)
df = df[safe_mask].copy()
print(f"After blocklist filter: {len(df):,}")

# ── Deduplicate — keep highest net_votes per (word, definition) pair ──────────
df = df.sort_values("net_votes", ascending=False)
df = df.drop_duplicates(subset=["word", "definition"])
print(f"After deduplication: {len(df):,}")

# ── Select output columns ─────────────────────────────────────────────────────
out = df[["word_id", "word", "definition", "up_votes", "down_votes",
          "net_votes", "upvote_ratio"]].copy()

out.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved {len(out):,} entries → {OUTPUT_PATH}")
print(f"\nSample:")
print(out.head(5).to_string(index=False))
