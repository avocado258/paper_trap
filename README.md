# PaperTrap: AI-Generated Research Paper Detection System

## Setup
```bash
pip install -r requirements.txt
```

---

## Data Collection

### Story 1: ArXiv Paper Download (Real Papers)
- Download the Kaggle ArXiv metadata snapshot from:
  https://www.kaggle.com/datasets/Cornell-University/arxiv
- Place `arxiv-metadata-oai-snapshot.json` in `data/`
- Run:
```bash
python src/data_collection/arxiv_downloader.py
```
- Output: 250 PDFs in `data/real_papers/`, metadata in `data/real_papers_metadata.csv`

---

### Story 2: AI-Generated Fake Paper Generation

Fake papers are generated using the **Gemini 3.1 Flash Lite Preview** API across three group members,
each responsible for a separate range of paper IDs.

#### Prerequisites
1. Create a free Gemini API key at https://aistudio.google.com/app/apikey
   - Sign in with a **personal Gmail account** (not a university account)
   - Click **"Create API key"** → **"Create new project"**
2. Create a `.env` file in the project root:
```
GEMINI_API_KEY=YOUR_KEY_HERE
```
3. Install the required package:
```bash
pip install google-genai python-dotenv
```

#### Member Assignments
| Member | Paper IDs | Provider |
|--------|-----------|----------|
| Member 1 | 1 – 84 | Gemini 2.0 Flash |
| Member 2 | 85 – 168 | Gemini 2.0 Flash |
| Member 3 | 169 – 250 | Gemini 2.0 Flash |

Each member only needs their own Gemini API key in their local `.env` file.

#### Running the Generator
```bash
# Member 1
python src/data_collection/paper_generator.py --member 1

# Member 2
python src/data_collection/paper_generator.py --member 2

# Member 3
python src/data_collection/paper_generator.py --member 3
```

#### Resume Support
The script is crash-safe. If generation is interrupted, simply re-run the same command —
it will automatically skip already-generated papers and continue from where it left off.
Progress is tracked by checking which `.txt` files already exist in `data/fake_papers/`.

#### Output
- `data/fake_papers/fake_paper_001.txt` … `fake_paper_250.txt` — one `.txt` file per paper
- `data/fake_papers_metadata.csv` — metadata for all generated papers with columns:
  `id, filename, label, provider, domain, topic, word_count, timestamp`
- `data/fake_papers_generation.log` — full generation log with warnings and errors

#### Notes
- Gemini 3.1 Flash Lite Preview free tier: **1,500 requests/day** — sufficient for each member's 84-paper range in a single session
- If you hit a 429 rate limit error, the script will automatically retry up to 3 times with exponential backoff
- Do **not** commit your `.env` file — it is listed in `.gitignore`
- Each paper is generated with randomized style, citation density, and heading format parameters to ensure stylometric diversity across the dataset