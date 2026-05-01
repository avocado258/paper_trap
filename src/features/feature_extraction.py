"""
PaperTrap — Feature Extraction Pipeline
========================================
Input:  final_dataset.csv  (columns: text, label)
Output: features.csv         (one row per paper, ~15 features + label)

All features are computed on the first 3,000 words of each document
to eliminate length-based confounds (AI papers are ~4x shorter than
human papers in this dataset).

Dependencies:
    pip install pandas numpy nltk spacy textstat transformers torch lexicalrichness
    python -m spacy download en_core_web_sm
    python -m nltk.downloader punkt averaged_perceptron_tagger stopwords
"""

import re
import string
import warnings
import numpy as np
import pandas as pd
import nltk
import spacy
import textstat
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from lexicalrichness import LexicalRichness

warnings.filterwarnings("ignore")

# ── NLTK downloads (safe to run repeatedly) ──────────────────────────────────
for pkg in ["punkt", "averaged_perceptron_tagger", "stopwords", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

# ── spaCy model ───────────────────────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm", disable=["ner"])
nlp.max_length = 2_000_000

# ── GPT-2 for perplexity ──────────────────────────────────────────────────────
print("Loading GPT-2 for perplexity scoring...")
_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
_model     = GPT2LMHeadModel.from_pretrained("gpt2")
_model.eval()
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model.to(_device)

STOP_WORDS = set(stopwords.words("english"))

# ─────────────────────────────────────────────────────────────────────────────
# 0. PREPROCESSING — truncate to first 3,000 words
# ─────────────────────────────────────────────────────────────────────────────

def truncate_to_words(text: str, n: int = 3000) -> str:
    """Return the first n whitespace-delimited words of text."""
    words = text.split()
    return " ".join(words[:n])


# ─────────────────────────────────────────────────────────────────────────────
# 1. LINGUISTIC FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_perplexity(text: str, max_tokens: int = 1024) -> float:
    """
    GPT-2 perplexity on the text window.
    Lower perplexity  → more predictable → more likely AI-generated.
    Truncated to max_tokens to fit GPT-2 context window.
    """
    encodings = _tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=max_tokens)
    input_ids = encodings.input_ids.to(_device)
    with torch.no_grad():
        loss = _model(input_ids, labels=input_ids).loss
    return float(torch.exp(loss).cpu())


def compute_mtld(text: str) -> float:
    """
    Measure of Textual Lexical Diversity — length-invariant vocabulary richness.
    Falls back to 0.0 if text is too short for MTLD calculation.
    """
    try:
        lex = LexicalRichness(text)
        if lex.words < 50:
            return 0.0
        return lex.mtld(threshold=0.72)
    except Exception:
        return 0.0


def compute_flesch(text: str) -> float:
    """Flesch Reading Ease score (higher = easier to read)."""
    score = textstat.flesch_reading_ease(text)
    # Clamp to valid range [0, 100]; scores outside this are PDF artifacts
    return max(0.0, min(100.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# 2. STATISTICAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_burstiness(text: str, max_sentences: int = 100) -> tuple[float, float]:
    """
    Sentence length variance computed on the first max_sentences sentences.
    Returns (mean_sent_len, std_sent_len).
    High std → bursty human writing; low std → uniform LLM output.
    """
    sentences = sent_tokenize(text)[:max_sentences]
    lengths   = [len(word_tokenize(s)) for s in sentences]
    if len(lengths) < 2:
        return 0.0, 0.0
    return float(np.mean(lengths)), float(np.std(lengths))


def compute_word_length_mean(text: str) -> float:
    """Mean character length of words (punctuation stripped)."""
    tokens = [t for t in word_tokenize(text) if t not in string.punctuation]
    if not tokens:
        return 0.0
    return float(np.mean([len(t) for t in tokens]))


def compute_function_word_ratio(text: str) -> float:
    """
    Fraction of tokens that are function words
    (articles, prepositions, conjunctions) — approximated via stopwords.
    """
    tokens = [t.lower() for t in word_tokenize(text)
              if t not in string.punctuation]
    if not tokens:
        return 0.0
    fw = sum(1 for t in tokens if t in STOP_WORDS)
    return fw / len(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# 3. STYLOMETRIC FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_pos_distributions(doc) -> dict:
    """
    Fraction of tokens belonging to each major POS category.
    Returns a dict: {pos_NOUN: float, pos_VERB: float, ...}
    """
    pos_tags  = [token.pos_ for token in doc if not token.is_space]
    total     = len(pos_tags) or 1
    counts    = Counter(pos_tags)
    categories = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP",
                  "CCONJ", "SCONJ", "NUM", "PUNCT"]
    return {f"pos_{p}": counts.get(p, 0) / total for p in categories}


def compute_passive_voice_rate(doc) -> float:
    """
    Passive voice frequency as a rate per sentence.
    Detected via spaCy dependency labels (nsubjpass / auxpass).
    """
    sentences     = list(doc.sents)
    n_sentences   = len(sentences) or 1
    passive_count = 0
    for sent in sentences:
        deps = {token.dep_ for token in sent}
        if "nsubjpass" in deps or "auxpass" in deps:
            passive_count += 1
    return passive_count / n_sentences


def compute_ngram_repetition_rate(text: str) -> dict:
    """
    Bigram and trigram repetition rates, normalized by token count.
    Rate = (number of repeated n-gram types) / total_tokens
    LLMs tend to repeat n-grams at higher rates than humans.
    """
    tokens  = [t.lower() for t in word_tokenize(text)
               if t not in string.punctuation]
    total   = len(tokens) or 1

    bigrams  = list(zip(tokens, tokens[1:]))
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))

    def repeated_types(ngrams):
        c = Counter(ngrams)
        return sum(1 for v in c.values() if v > 1)

    return {
        "bigram_repetition_rate":  repeated_types(bigrams)  / total,
        "trigram_repetition_rate": repeated_types(trigrams) / total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. STRUCTURAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_citation_density(text: str) -> float:
    """
    Citations per 1,000 words in the fixed window.
    Detects patterns like [1], [1,2], (Author, 2020), [Author et al., 2021].
    """
    patterns = [
        r"\[\d+(?:,\s*\d+)*\]",           # [1], [1,2,3]
        r"\(\w[\w\s]+,\s*\d{4}\)",         # (Author, 2020)
        r"\[\w[\w\s]+et\s+al\.,?\s*\d{4}\]",  # [Author et al., 2021]
    ]
    combined  = "|".join(patterns)
    citations = len(re.findall(combined, text))
    words     = len(text.split()) or 1
    return (citations / words) * 1000


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN EXTRACTION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(text: str) -> dict:
    """Extract all features from a single (already-truncated) text."""
    doc = nlp(text)

    features = {}

    # — Linguistic
    features["perplexity"]    = compute_perplexity(text)
    features["mtld"]          = compute_mtld(text)
    features["flesch_ease"]   = compute_flesch(text)

    # — Statistical
    mean_sl, std_sl = compute_burstiness(text)
    features["sent_len_mean"]          = mean_sl
    features["sent_len_std"]           = std_sl
    features["word_len_mean"]          = compute_word_length_mean(text)
    features["function_word_ratio"]    = compute_function_word_ratio(text)

    # — Stylometric
    features.update(compute_pos_distributions(doc))
    features["passive_voice_rate"]     = compute_passive_voice_rate(doc)
    features.update(compute_ngram_repetition_rate(text))

    # — Structural
    features["citation_density"]       = compute_citation_density(text)

    return features


def run_pipeline(input_csv: str, output_csv: str, word_limit: int = 3000):
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} papers. Extracting features...")

    records = []
    for idx, row in df.iterrows():
        truncated = truncate_to_words(row["text"], word_limit)
        try:
            feats = extract_features(truncated)
        except Exception as e:
            print(f"  [WARN] Row {idx} failed: {e}")
            feats = {}
        feats["label"] = row["label"]
        records.append(feats)

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(df)}")

    out = pd.DataFrame(records)

    # encode label: AI=1, Human=0
    out["label"] = out["label"].map({"AI": 1, "Human": 0})

    out.to_csv(output_csv, index=False)
    print(f"\nDone. Features saved to: {output_csv}")
    print(f"Shape: {out.shape}")
    print(f"Columns: {list(out.columns)}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    INPUT_CSV  = "../../data/final_dataset.csv"
    OUTPUT_CSV = "../../outputs/features.csv"
    run_pipeline(INPUT_CSV, OUTPUT_CSV)