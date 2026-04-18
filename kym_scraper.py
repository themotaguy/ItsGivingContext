"""
Know Your Meme scraper for the slang + meme RAG project.
Crawls confirmed/notable meme pages and saves structured text data to CSV.

Strategy:
  1. Paginate the /memes?sort=reverse-chronological listing pages to collect slugs
  2. Fetch each /memes/<slug> page and parse structured fields + body sections
  3. Save incrementally so crashes don't lose progress

Requirements:
    pip install requests beautifulsoup4 pandas tqdm

Be polite: default delay is 1.5s between requests. Don't reduce below 1s.
"""

import csv
import re
import time
import random
import logging
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL        = "https://knowyourmeme.com"
OUTPUT_PATH     = Path("data/know_your_meme.csv")
CHECKPOINT_PATH = Path("data/kym_scraped_slugs.txt")   # tracks already-scraped slugs
TARGET_COUNT    = 15_000          # aim for more than 10k to have buffer after filtering
KEEP_STATUSES   = {"confirmed", "notable"}
MIN_DELAY       = 3.0             # seconds between requests — be polite
MAX_DELAY       = 6.0
MAX_RETRIES     = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; RAG-research-bot/1.0; "
        "Northeastern University NLP class project; polite crawler)"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── HTTP helpers ──────────────────────────────────────────────────────────────
session = requests.Session()
session.headers.update(HEADERS)

def fetch(url: str, retries: int = MAX_RETRIES) -> BeautifulSoup | None:
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
                return BeautifulSoup(resp.text, "html.parser")
            elif resp.status_code == 429:
                wait = 60 * (attempt + 1)
                log.warning(f"Rate limited — waiting {wait}s")
                time.sleep(wait)
            elif resp.status_code == 404:
                return None
            else:
                log.warning(f"HTTP {resp.status_code} for {url}")
                time.sleep(5)
        except requests.RequestException as e:
            log.error(f"Request error ({attempt+1}/{retries}): {e}")
            time.sleep(10)
    return None

# ── Step 1: Collect meme slugs from listing pages ─────────────────────────────
def collect_slugs(max_slugs: int) -> list[str]:
    """
    Paginate /memes?sort=reverse-chronological to collect meme page slugs.
    KYM listing pages use ?page=N with 24 entries per page.
    """
    slugs = []
    page  = 1

    log.info(f"Collecting up to {max_slugs} slugs from listing pages...")
    with tqdm(total=max_slugs, desc="Collecting slugs") as pbar:
        while len(slugs) < max_slugs:
            url  = f"{BASE_URL}/memes?sort=reverse-chronological&page={page}"
            soup = fetch(url)
            if not soup:
                log.warning(f"Failed to fetch listing page {page}, stopping.")
                break

            # Current KYM structure: <li class="index"><a href="/memes/slug">...</a></li>
            entries = soup.select("li.index a[href^='/memes/']")
            if not entries:
                # fallback: any anchor linking to a top-level /memes/slug
                entries = [
                    a for a in soup.select("a[href^='/memes/']")
                    if a.get("href", "").count("/") == 2  # exactly /memes/slug
                ]

            if not entries:
                log.info(f"No entries found on page {page} — likely end of listings.")
                break

            new = [
                a["href"].replace("/memes/", "").strip("/")
                for a in entries
                if a.get("href", "").startswith("/memes/")
                and a["href"].replace("/memes/", "").strip("/")
                and "/" not in a["href"].replace("/memes/", "").strip("/")
            ]
            slugs.extend(new)
            pbar.update(len(new))
            page += 1

    return list(dict.fromkeys(slugs))  # deduplicate, preserve order

# ── Step 2: Parse a single meme page ─────────────────────────────────────────
def parse_sections(soup: BeautifulSoup) -> dict[str, str]:
    """Extract named sections using KYM's anchor IDs (about, origin, spread, etc.)."""
    sections = {}
    target_sections = ["about", "origin", "spread", "notable-examples"]
    section_keys    = ["about", "origin", "spread", "notable examples"]

    for anchor_id, key in zip(target_sections, section_keys):
        # KYM marks each section with an anchor: <h2 id="about"> or <a id="about">
        heading = soup.find(id=anchor_id)
        if not heading:
            continue

        # Collect all text nodes until the next heading
        texts = []
        for sibling in heading.find_all_next():
            tag = getattr(sibling, "name", None)
            if tag in ("h1", "h2", "h3") and sibling != heading:
                break
            # Skip nested headings and navigation elements
            if tag in ("nav", "aside", "script", "style"):
                continue
            if tag in ("p", "li", "blockquote", "div") and sibling.find_parent(["nav", "aside"]) is None:
                text = sibling.get_text(" ", strip=True)
                if text and len(text) > 20:
                    texts.append(text)

        body_text = " ".join(texts).strip()
        if len(body_text) > 50:
            sections[key] = body_text

    return sections

def parse_sidebar(soup: BeautifulSoup) -> dict:
    """Extract structured metadata from the sidebar (dl/dt/dd structure)."""
    meta = {}

    # Current KYM structure: dl with dt/dd pairs
    for dl in soup.select("dl"):
        dts = dl.select("dt")
        dds = dl.select("dd")
        for dt, dd in zip(dts, dds):
            key = dt.get_text(strip=True).lower().replace(" ", "_")
            val = dd.get_text(" ", strip=True)
            if key and val:
                meta[key] = val

    # Fallback: old table structure
    if not meta:
        for row in soup.select("tr"):
            cells = row.select("td, th")
            if len(cells) >= 2:
                key = cells[0].get_text(strip=True).lower().replace(" ", "_")
                val = cells[1].get_text(" ", strip=True)
                meta[key] = val

    return meta

def parse_status(soup: BeautifulSoup) -> str:
    """Extract the entry status (confirmed, notable, etc.)."""
    # Current KYM structure: <div class="status"><span>Confirmed</span></div>
    el = soup.select_one("div.status span, div.status")
    if el:
        return el.get_text(strip=True).lower()

    # Fallback: look for dt with text "status" and read sibling dd
    for dt in soup.select("dt"):
        if dt.get_text(strip=True).lower() == "status":
            dd = dt.find_next_sibling("dd")
            if dd:
                return dd.get_text(strip=True).lower()

    # Last resort: scan page text
    text = soup.get_text(" ", strip=True).lower()
    for status in ("confirmed", "notable", "newsworthy", "deadpool", "submission"):
        if f"status {status}" in text:
            return status

    return "unknown"

def parse_meme_page(slug: str) -> dict | None:
    url  = f"{BASE_URL}/memes/{slug}"
    soup = fetch(url)
    if not soup:
        return None

    # Title — use og:title meta tag (most reliable), fallback to slug
    og_title = soup.select_one("meta[property='og:title']")
    if og_title and og_title.get("content"):
        raw = og_title["content"]
        # Strip " | Know Your Meme" suffix if present
        title = raw.split(" | ")[0].strip()
    else:
        title = slug.replace("-", " ").title()

    # Status
    status = parse_status(soup)

    # Tags
    tag_els = soup.select("a.tag, ul.entry-tags a, .tag-list a, a[href*='/tags/']")
    tags = ", ".join(t.get_text(strip=True) for t in tag_els if t.get_text(strip=True))

    # Sidebar metadata
    sidebar = parse_sidebar(soup)
    year = None
    for key in ("year", "origin_year", "added"):
        val = sidebar.get(key, "")
        m = re.search(r"\b(19|20)\d{2}\b", val)
        if m:
            year = int(m.group())
            break

    platform = sidebar.get("origin", sidebar.get("type", ""))

    # Body sections
    sections = parse_sections(soup)

    return {
        "slug":            slug,
        "title":           title,
        "status":          status,
        "tags":            tags,
        "year":            year,
        "platform":        platform,
        "url":             url,
        "body_about":      sections.get("about", ""),
        "body_origin":     sections.get("origin", ""),
        "body_spread":     sections.get("spread", ""),
        "body_examples":   sections.get("notable examples", ""),
    }

# ── Step 3: Incremental save ──────────────────────────────────────────────────
FIELDNAMES = [
    "slug", "title", "status", "tags", "year", "platform", "url",
    "body_about", "body_origin", "body_spread", "body_examples",
]

def load_checkpoint() -> set[str]:
    if CHECKPOINT_PATH.exists():
        return set(CHECKPOINT_PATH.read_text().splitlines())
    return set()

def save_checkpoint(slug: str):
    with open(CHECKPOINT_PATH, "a") as f:
        f.write(slug + "\n")

def append_row(row: dict):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUTPUT_PATH.exists()
    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    already_scraped = load_checkpoint()
    log.info(f"Resuming — {len(already_scraped)} slugs already done.")

    # Collect more slugs than needed to have buffer after status filtering
    slugs = collect_slugs(max_slugs=TARGET_COUNT + len(already_scraped) + 2000)
    slugs = [s for s in slugs if s not in already_scraped]
    log.info(f"{len(slugs)} new slugs to scrape.")

    saved = 0
    for slug in tqdm(slugs, desc="Scraping meme pages"):
        result = parse_meme_page(slug)

        if result is None:
            save_checkpoint(slug)
            continue

        # Filter by status
        if result["status"] not in KEEP_STATUSES:
            save_checkpoint(slug)
            continue

        # Filter out entries with no meaningful body text
        body_total = sum(len(result.get(f, "")) for f in
                         ["body_about", "body_origin", "body_spread", "body_examples"])
        if body_total < 100:
            save_checkpoint(slug)
            continue

        append_row(result)
        save_checkpoint(slug)
        saved += 1

        if saved % 500 == 0:
            log.info(f"Saved {saved} entries so far.")

    log.info(f"Done. Total saved: {saved} entries → {OUTPUT_PATH}")

    # Quick summary
    if OUTPUT_PATH.exists():
        df = pd.read_csv(OUTPUT_PATH)
        print(f"\nFinal dataset: {len(df)} rows")
        print(df["status"].value_counts())
        print(f"Avg body_origin length: {df['body_origin'].str.len().mean():.0f} chars")