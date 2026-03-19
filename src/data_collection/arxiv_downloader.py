"""
PaperTrap - ArXiv Paper Downloader
Uses Kaggle ArXiv metadata snapshot to filter papers,
then downloads PDFs directly via arxiv.org URLs using requests.

Kaggle dataset: https://www.kaggle.com/datasets/Cornell-University/arxiv
Download the dataset and place 'arxiv-metadata-oai-snapshot.json' in data/
"""

import os
import csv
import time
import json
import logging
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

METADATA_JSON = "data/arxiv-metadata-oai-snapshot.json"  # Kaggle snapshot
OUTPUT_DIR    = "data/real_papers"
METADATA_CSV  = "data/real_papers_metadata.csv"

TARGET_CATEGORIES = {"cs.AI", "cs.LG", "cs.CL"}
DATE_START = "2020"
DATE_END   = "2024"
TARGET_COUNT = 250

SLEEP_BETWEEN_DOWNLOADS = 1   # seconds
MAX_RETRIES = 3

HEADERS = {
    "User-Agent": "PaperTrap-Research/1.0 (academic project)"
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def matches_filter(entry: dict) -> bool:
    """Return True if entry matches category and date filters."""
    # Categories field is space-separated string e.g. "cs.LG cs.AI stat.ML"
    cats = set(entry.get("categories", "").split())
    if not cats.intersection(TARGET_CATEGORIES):
        return False

    # update_date format: "YYYY-MM-DD"
    year = entry.get("update_date", "")[:4]
    if not (DATE_START <= year <= DATE_END):
        return False

    return True


def arxiv_id_to_url(arxiv_id: str) -> str:
    """Convert ArXiv ID to direct PDF URL."""
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def download_pdf(arxiv_id: str, dest_dir: str) -> str | None:
    """Download a single PDF. Returns filepath or None on failure."""
    filename = f"{arxiv_id.replace('/', '_')}.pdf"
    filepath = os.path.join(dest_dir, filename)

    if os.path.exists(filepath):
        logger.info(f"  [SKIP] Already exists: {filename}")
        return filepath

    url = arxiv_id_to_url(arxiv_id)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30, stream=True)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"  [OK] {filename}")
                return filepath
            else:
                logger.warning(f"  [HTTP {response.status_code}] {filename} — attempt {attempt}/{MAX_RETRIES}")
        except requests.RequestException as e:
            logger.warning(f"  [RETRY {attempt}/{MAX_RETRIES}] {filename} — {e}")

        time.sleep(SLEEP_BETWEEN_DOWNLOADS * attempt)

    logger.error(f"  [FAIL] {filename}")
    return None


def load_filtered_papers(json_path: str, target: int) -> list[dict]:
    """
    Stream-parse the Kaggle JSON snapshot (one JSON object per line)
    and return up to `target` filtered entries, balanced across categories.
    """
    logger.info(f"Scanning metadata snapshot: {json_path}")
    filtered = []

    per_category = {cat: 0 for cat in TARGET_CATEGORIES}
    per_category_target = (target + 20) // len(TARGET_CATEGORIES)  # ~83 each, but buffer for failures

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(filtered) >= target + 20: # collect 250 + 20 candidates
                break
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            if not matches_filter(entry):
                continue

            cats = set(entry.get("categories", "").split()).intersection(TARGET_CATEGORIES)
            eligible_cats = [c for c in cats if per_category[c] < per_category_target]
            if not eligible_cats:
                continue

            primary_cat = eligible_cats[0]
            per_category[primary_cat] += 1
            entry["_matched_category"] = primary_cat
            filtered.append(entry)

    logger.info(f"Filtered {len(filtered)} papers. Per-category: {per_category}")
    return filtered


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    papers = load_filtered_papers(METADATA_JSON, TARGET_COUNT + 20)

    if not papers:
        logger.error("No papers found. Check metadata path and filters.")
        return

    metadata_rows = []
    total_downloaded = 0

    for i, entry in enumerate(papers, 1):
        arxiv_id = entry.get("id", "").strip()
        if not arxiv_id:
            continue

        logger.info(f"[{i}/{len(papers)}] {arxiv_id}")
        filepath = download_pdf(arxiv_id, OUTPUT_DIR)

        if filepath:
            metadata_rows.append({
                "arxiv_id":   arxiv_id,
                "title":      entry.get("title", "").replace("\n", " "),
                "authors":    entry.get("authors", "").replace("\n", " "),
                "abstract":   entry.get("abstract", "").replace("\n", " "),
                "category":   entry.get("_matched_category", ""),
                "published":  entry.get("update_date", ""),
                "pdf_path":   filepath,
                "label":      "real",
            })
            total_downloaded += 1
            if total_downloaded >= TARGET_COUNT:
                break

        time.sleep(SLEEP_BETWEEN_DOWNLOADS)

    # ── Save metadata ──────────────────────────────────────────────────────────
    fieldnames = ["arxiv_id", "title", "authors", "abstract", "category",
                  "published", "pdf_path", "label"]

    with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    logger.info(f"\n✓ Done. Total downloaded: {total_downloaded}/{len(papers)}")
    logger.info(f"✓ Metadata saved to: {METADATA_CSV}")


if __name__ == "__main__":
    main()