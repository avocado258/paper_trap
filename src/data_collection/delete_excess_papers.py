import os
import csv

CSV_PATH = "data/real_papers_metadata.csv"
PDF_DIR  = "data/real_papers"

with open(CSV_PATH, "r", encoding="utf-8") as f:
    valid_ids = {row["arxiv_id"].replace("/", "_") + ".pdf" for row in csv.DictReader(f)}

for filename in os.listdir(PDF_DIR):
    if filename not in valid_ids:
        os.remove(os.path.join(PDF_DIR, filename))
        print(f"Deleted: {filename}")