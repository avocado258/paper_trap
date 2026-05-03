"""
ml/shap_explainer.py
────────────────────
SHAP Explainability module for PaperTrap.

Real usage  : swap compute_shap_explanation() body to load your actual
              trained ensemble (.pkl) and call shap.TreeExplainer on it.

Demo mode   : returns realistic synthetic SHAP values so the dashboard
              works end-to-end without a trained model present.
"""

from __future__ import annotations
import random
import math
from dataclasses import dataclass, field


# ── Data shapes ───────────────────────────────────────────────────────────────

@dataclass
class ShapFeature:
    name:        str
    value:       float          # raw feature value extracted from the paper
    shap_value:  float          # SHAP attribution (+ → pushes AI, - → pushes human)
    category:    str            # linguistic | statistical | stylometric | structural
    direction:   str = field(init=False)

    def __post_init__(self):
        self.direction = "positive" if self.shap_value >= 0 else "negative"


@dataclass
class ShapResult:
    verdict:          str              # "AI-Generated" | "Human-Authored"
    confidence:       float            # 0–100
    base_value:       float            # model's prior (≈ 0.5 for balanced dataset)
    features:         list[ShapFeature]
    category_scores:  dict[str, float] # per-category risk 0–1
    model_votes:      dict[str, float] # RF / GB / Ensemble


# ── Feature catalogue (all 60 signals) ───────────────────────────────────────

FEATURE_CATALOGUE: list[dict] = [
    # ── Linguistic (18) ──────────────────────────────────────────────────────
    {"name": "Perplexity Score",          "category": "linguistic",   "ai_low": True },
    {"name": "Type-Token Ratio (TTR)",    "category": "linguistic",   "ai_low": True },
    {"name": "Flesch-Kincaid Grade",      "category": "linguistic",   "ai_low": False},
    {"name": "Avg Sentence Length",       "category": "linguistic",   "ai_low": False},
    {"name": "Sentence Length Variance",  "category": "linguistic",   "ai_low": True },
    {"name": "Hapax Legomena Rate",       "category": "linguistic",   "ai_low": True },
    {"name": "Vocabulary Richness (MTLD)","category": "linguistic",   "ai_low": True },
    {"name": "Discourse Coherence",       "category": "linguistic",   "ai_low": False},
    {"name": "Avg Word Length",           "category": "linguistic",   "ai_low": False},
    {"name": "Lexical Density",           "category": "linguistic",   "ai_low": True },
    {"name": "Readability (ARI)",         "category": "linguistic",   "ai_low": False},
    {"name": "Syllable Complexity",       "category": "linguistic",   "ai_low": False},
    {"name": "Connective Density",        "category": "linguistic",   "ai_low": False},
    {"name": "Hedging Rate",              "category": "linguistic",   "ai_low": False},
    {"name": "Negation Frequency",        "category": "linguistic",   "ai_low": True },
    {"name": "Modal Verb Ratio",          "category": "linguistic",   "ai_low": False},
    {"name": "Passive Voice Rate",        "category": "linguistic",   "ai_low": False},
    {"name": "Coreference Density",       "category": "linguistic",   "ai_low": True },
    # ── Statistical (14) ─────────────────────────────────────────────────────
    {"name": "Bigram Entropy",            "category": "statistical",  "ai_low": True },
    {"name": "Trigram Entropy",           "category": "statistical",  "ai_low": True },
    {"name": "Unigram Skew",              "category": "statistical",  "ai_low": True },
    {"name": "Burstiness Coefficient",    "category": "statistical",  "ai_low": True },
    {"name": "Zipf's Law Deviation",      "category": "statistical",  "ai_low": True },
    {"name": "Character Entropy",         "category": "statistical",  "ai_low": True },
    {"name": "Punctuation Density",       "category": "statistical",  "ai_low": True },
    {"name": "Token Frequency Skew",      "category": "statistical",  "ai_low": True },
    {"name": "Sentence Entropy",          "category": "statistical",  "ai_low": True },
    {"name": "Repetition Index",          "category": "statistical",  "ai_low": False},
    {"name": "N-gram Overlap Rate",       "category": "statistical",  "ai_low": False},
    {"name": "Vocabulary Growth Rate",    "category": "statistical",  "ai_low": True },
    {"name": "Long-tail Frequency",       "category": "statistical",  "ai_low": True },
    {"name": "Symbol Density",            "category": "statistical",  "ai_low": True },
    # ── Stylometric (16) ─────────────────────────────────────────────────────
    {"name": "Noun Ratio",                "category": "stylometric",  "ai_low": False},
    {"name": "Verb Ratio",                "category": "stylometric",  "ai_low": True },
    {"name": "Adjective Ratio",           "category": "stylometric",  "ai_low": False},
    {"name": "Adverb Ratio",              "category": "stylometric",  "ai_low": True },
    {"name": "Function Word Ratio",       "category": "stylometric",  "ai_low": True },
    {"name": "Stopword Ratio",            "category": "stylometric",  "ai_low": False},
    {"name": "Dependency Depth",          "category": "stylometric",  "ai_low": True },
    {"name": "Clause Complexity",         "category": "stylometric",  "ai_low": True },
    {"name": "Subordination Rate",        "category": "stylometric",  "ai_low": True },
    {"name": "Coordination Rate",         "category": "stylometric",  "ai_low": False},
    {"name": "Prepositional Phrase Rate", "category": "stylometric",  "ai_low": False},
    {"name": "Named Entity Density",      "category": "stylometric",  "ai_low": True },
    {"name": "POS Trigram Entropy",       "category": "stylometric",  "ai_low": True },
    {"name": "Sentence Variety Score",    "category": "stylometric",  "ai_low": True },
    {"name": "Apposition Rate",           "category": "stylometric",  "ai_low": True },
    {"name": "Relative Clause Rate",      "category": "stylometric",  "ai_low": True },
    # ── Structural (12) ──────────────────────────────────────────────────────
    {"name": "Citation Density",          "category": "structural",   "ai_low": True },
    {"name": "Abstract/Body Ratio",       "category": "structural",   "ai_low": False},
    {"name": "Intro/Conclusion Ratio",    "category": "structural",   "ai_low": False},
    {"name": "Section Heading Count",     "category": "structural",   "ai_low": False},
    {"name": "Avg Paragraph Length",      "category": "structural",   "ai_low": False},
    {"name": "Reference List Length",     "category": "structural",   "ai_low": True },
    {"name": "Figure/Table Mentions",     "category": "structural",   "ai_low": True },
    {"name": "Equation Density",          "category": "structural",   "ai_low": True },
    {"name": "Footnote Rate",             "category": "structural",   "ai_low": True },
    {"name": "Acknowledgement Presence",  "category": "structural",   "ai_low": True },
    {"name": "Cross-Reference Density",   "category": "structural",   "ai_low": True },
    {"name": "URL/DOI Density",           "category": "structural",   "ai_low": True },
]


# ── Demo mode: generate realistic synthetic SHAP values ───────────────────────

def _synthetic_feature_value(spec: dict, is_ai: bool) -> tuple[float, float]:
    """Return (raw_value, shap_value) for demo mode."""
    rng = random.Random(hash(spec["name"]) % 9999)

    if spec["category"] == "linguistic":
        raw = rng.uniform(12, 30) if is_ai else rng.uniform(55, 120)
        if spec["name"] == "Type-Token Ratio (TTR)":
            raw = rng.uniform(0.32, 0.48) if is_ai else rng.uniform(0.60, 0.82)
        if spec["name"] == "Flesch-Kincaid Grade":
            raw = rng.uniform(65, 80) if is_ai else rng.uniform(38, 58)

    elif spec["category"] == "statistical":
        raw = rng.uniform(2.1, 3.4) if is_ai else rng.uniform(4.0, 6.2)

    elif spec["category"] == "stylometric":
        raw = rng.uniform(0.12, 0.28) if is_ai else rng.uniform(0.35, 0.62)

    else:  # structural
        raw = rng.uniform(0.01, 0.08) if is_ai else rng.uniform(0.12, 0.45)

    # SHAP value: how much this feature pushed the prediction toward AI
    magnitude = rng.uniform(0.04, 0.45)
    # If ai_low=True, lower value → AI → positive SHAP contribution
    if spec["ai_low"]:
        shap = +magnitude if is_ai else -magnitude * 0.4
    else:
        shap = -magnitude * 0.4 if is_ai else +magnitude

    # Add small noise
    shap += rng.uniform(-0.02, 0.02)
    return round(raw, 3), round(shap, 4)


def compute_shap_explanation(text: str, filename: str = "paper.pdf") -> ShapResult:
    """
    Main entry point.

    In production: load your .pkl model, extract real features, run
    shap.TreeExplainer(model).shap_values(feature_matrix).

    In demo mode: returns synthetic but realistic values.
    """
    # ── Determine verdict from simple heuristics on text ─────────────────
    words      = text.split() if text else []
    word_count = len(words)

    if word_count < 50:
        # Not enough text → treat as borderline AI for demo
        confidence = 65.0
        is_ai      = True
    else:
        unique_ratio = len(set(words)) / word_count
        avg_word_len = sum(len(w) for w in words) / word_count
        # Simple heuristic: low uniqueness + short avg word = AI-like
        ai_score = (1 - unique_ratio) * 60 + max(0, (5.5 - avg_word_len)) * 8
        confidence = min(98, max(15, ai_score + random.uniform(-5, 5)))
        is_ai      = confidence >= 50

    verdict = "AI-Generated" if is_ai else "Human-Authored"

    # ── Build feature list ────────────────────────────────────────────────
    features: list[ShapFeature] = []
    for spec in FEATURE_CATALOGUE:
        raw, shap_val = _synthetic_feature_value(spec, is_ai)
        features.append(ShapFeature(
            name       = spec["name"],
            value      = raw,
            shap_value = shap_val,
            category   = spec["category"],
        ))

    # Sort by |shap_value| descending
    features.sort(key=lambda f: abs(f.shap_value), reverse=True)

    # ── Category risk scores ──────────────────────────────────────────────
    category_scores: dict[str, float] = {}
    for cat in ("linguistic", "statistical", "stylometric", "structural"):
        cat_feats = [f for f in features if f.category == cat]
        pos = sum(f.shap_value for f in cat_feats if f.shap_value > 0)
        total = sum(abs(f.shap_value) for f in cat_feats) or 1
        category_scores[cat] = round(min(1.0, pos / total), 3)

    # ── Ensemble votes ────────────────────────────────────────────────────
    noise = random.uniform(-3, 3)
    rf_conf = round(min(99, max(5, confidence + noise)), 1)
    gb_conf = round(min(99, max(5, confidence - noise + random.uniform(-2, 2))), 1)
    model_votes = {
        "Random Forest":     rf_conf,
        "Gradient Boosting": gb_conf,
        "Soft Voting":       round(confidence, 1),
    }

    return ShapResult(
        verdict         = verdict,
        confidence      = round(confidence, 1),
        base_value      = 0.5,
        features        = features,
        category_scores = category_scores,
        model_votes     = model_votes,
    )