# PaperTrap: AI-Generated Research Paper Detection System

## Setup
```bash
pip install -r requirements.txt
```

## Data Collection

### Story 1: ArXiv Paper Download
- Download the Kaggle ArXiv metadata snapshot from:
  https://www.kaggle.com/datasets/Cornell-University/arxiv
- Place `arxiv-metadata-oai-snapshot.json` in `data/`
- Run:
```bash
python src/data_collection/arxiv_downloader.py
```
- Output: 250 PDFs in `data/real_papers/`, metadata in `data/real_papers_metadata.csv`