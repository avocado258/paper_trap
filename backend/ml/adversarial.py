"""
ml/adversarial.py
─────────────────
Adversarial robustness test suite for PaperTrap.

Tests whether the detector is fooled by common evasion techniques:
  1. T5 Paraphrasing    — rewrite sentences while preserving meaning
  2. Synonym Swapping   — replace high-frequency words with synonyms
  3. Sentence Shuffling — reorder sentences within paragraphs
  4. SCIgen Baseline    — computer-generated nonsense paper comparison
  5. GPT-4 Rewrite      — stronger model attempt to bypass detection

Real usage : plug in actual T5/GPT-4 API calls and re-run your model.
Demo mode  : returns realistic synthetic results without any API calls.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from enum import Enum


# ── Enums & data shapes ───────────────────────────────────────────────────────

class TestStatus(str, Enum):
    PASSED  = "passed"    # detector still caught it after attack
    WARNING = "warning"   # confidence dropped significantly (>15pp)
    FAILED  = "failed"    # detector was fooled (verdict flipped or conf < 50)


@dataclass
class AttackResult:
    attack_name:        str
    attack_type:        str           # paraphrase | lexical | structural | baseline
    description:        str
    original_conf:      float         # confidence before attack (0-100)
    attacked_conf:      float         # confidence after attack
    confidence_drop:    float         # original - attacked
    verdict_flipped:    bool          # did the verdict change?
    status:             TestStatus
    robustness_score:   float         # 0-100 (100 = perfectly robust)
    highlighted_changes: list[str]    # example changed phrases
    technique_details:  str           # how the attack was done


@dataclass
class AdversarialReport:
    filename:            str
    original_verdict:    str
    original_confidence: float
    overall_robustness:  float          # weighted average across all attacks
    overall_status:      TestStatus
    results:             list[AttackResult]
    summary:             dict[str, int]  # passed/warning/failed counts
    recommendations:     list[str]


# ── Attack definitions ────────────────────────────────────────────────────────

ATTACKS: list[dict] = [
    {
        "name":        "T5 Paraphrasing",
        "type":        "paraphrase",
        "description": "Each sentence is paraphrased by a T5-base model while preserving semantic meaning. Tests whether surface-level rephrasing breaks perplexity and n-gram signals.",
        "technique":   "Google T5-base seq2seq paraphraser, beam search k=5, temperature=0.7. Applied sentence-by-sentence.",
        "severity":    "high",
        "conf_attack_range": (0.88, 0.97),   # multiplier on original conf
    },
    {
        "name":        "Synonym Swapping (WordNet)",
        "type":        "lexical",
        "description": "High-frequency content words are replaced with WordNet synonyms. Targets TTR (Type-Token Ratio) and vocabulary richness features.",
        "technique":   "NLTK WordNet synsets, top-30% frequency words replaced, excluding stop words and domain terms.",
        "severity":    "medium",
        "conf_attack_range": (0.90, 0.98),
    },
    {
        "name":        "Sentence Shuffling",
        "type":        "structural",
        "description": "Sentences within each paragraph are randomly reordered. Attacks discourse coherence and structural section-ratio features.",
        "technique":   "Paragraph boundary detection via spaCy, random permutation within each paragraph block.",
        "severity":    "low",
        "conf_attack_range": (0.93, 0.99),
    },
    {
        "name":        "Back-Translation (EN→DE→EN)",
        "type":        "paraphrase",
        "description": "Text is translated to German then back to English using Helsinki-NLP models. Introduces natural lexical variation while preserving content.",
        "technique":   "Helsinki-NLP/opus-mt-en-de → Helsinki-NLP/opus-mt-de-en pipeline via HuggingFace Transformers.",
        "severity":    "high",
        "conf_attack_range": (0.82, 0.95),
    },
    {
        "name":        "GPT-4 Humanisation Rewrite",
        "type":        "paraphrase",
        "description": "GPT-4 is prompted to rewrite the abstract and introduction to 'sound more like a human researcher.' The strongest known evasion technique.",
        "technique":   "OpenAI GPT-4-turbo with system prompt: 'Rewrite to increase perplexity and vocabulary diversity while preserving scientific content.'",
        "severity":    "critical",
        "conf_attack_range": (0.60, 0.88),
    },
    {
        "name":        "SCIgen Baseline Comparison",
        "type":        "baseline",
        "description": "Compares the paper's feature profile against known SCIgen-generated computer science papers (pure nonsense). Establishes a lower-bound sanity check.",
        "technique":   "Feature-distance comparison against 50 SCIgen papers in the test corpus. Reports Euclidean distance in 60-D feature space.",
        "severity":    "medium",
        "conf_attack_range": (0.94, 0.99),
    },
    {
        "name":        "Homoglyph Substitution",
        "type":        "lexical",
        "description": "Unicode homoglyphs replace Latin characters (e.g. 'а' Cyrillic for 'a'). Tests tokeniser robustness — invisible to human readers.",
        "technique":   "Random 5% of alphabetic characters replaced with visually identical Unicode codepoints from Cyrillic/Greek blocks.",
        "severity":    "low",
        "conf_attack_range": (0.95, 1.00),
    },
    {
        "name":        "Abstract Injection",
        "type":        "structural",
        "description": "The abstract is replaced with a human-written abstract from a real ArXiv paper on a related topic. Tests whether structural signals dominate.",
        "technique":   "Random sampling from ArXiv-Human test set (cs.LG category), abstract section replaced, body unchanged.",
        "severity":    "medium",
        "conf_attack_range": (0.85, 0.97),
    },
]


# ── Demo computation ──────────────────────────────────────────────────────────

_EXAMPLE_CHANGES: dict[str, list[str]] = {
    "paraphrase": [
        '"novel framework" → "innovative approach"',
        '"significant improvements" → "notable gains"',
        '"we propose" → "this paper introduces"',
        '"extensive experiments" → "thorough evaluations"',
    ],
    "lexical": [
        '"framework" → "architecture"',
        '"significant" → "substantial"',
        '"leverage" → "utilise"',
        '"demonstrate" → "exhibit"',
    ],
    "structural": [
        "Sentence order permuted within §3.2",
        "Paragraph 4 sentences reordered",
        "Introduction sentences shuffled",
    ],
    "baseline": [
        "Feature distance to SCIgen centroid: 3.84",
        "Closest SCIgen paper similarity: 0.41",
        "Above SCIgen perplexity floor: ✓",
    ],
}

_EXAMPLE_CHANGES["paraphrase"] += [
    '"deep learning" → "neural network approaches"',
    '"outperforms" → "achieves superior performance to"',
]


def run_adversarial_tests(
    text: str,
    filename: str,
    original_verdict: str,
    original_confidence: float,
) -> AdversarialReport:
    """
    Run all adversarial attacks and return a full report.

    Production: integrate real T5/GPT-4 API calls and re-run your
    trained ensemble on each mutated text.
    Demo mode: synthesises realistic results deterministically.
    """
    rng = random.Random(len(text) + int(original_confidence * 100))

    results: list[AttackResult] = []

    for attack in ATTACKS:
        lo, hi = attack["conf_attack_range"]
        multiplier    = rng.uniform(lo, hi)
        attacked_conf = round(original_confidence * multiplier, 1)
        conf_drop     = round(original_confidence - attacked_conf, 1)

        # Verdict flips only if attacked confidence drops below 50
        verdict_flipped = attacked_conf < 50.0 and original_verdict == "AI-Generated"

        # Status rules
        if verdict_flipped or attacked_conf < 50:
            status = TestStatus.FAILED
        elif conf_drop > 15:
            status = TestStatus.WARNING
        else:
            status = TestStatus.PASSED

        # Robustness score: 100 if unchanged, scales down with drop
        robustness = max(0.0, round(100 - conf_drop * 1.8, 1))

        # Pick example changes
        change_pool = _EXAMPLE_CHANGES.get(attack["type"], [])
        n_changes   = rng.randint(2, min(4, len(change_pool)))
        changes     = rng.sample(change_pool, n_changes)

        results.append(AttackResult(
            attack_name         = attack["name"],
            attack_type         = attack["type"],
            description         = attack["description"],
            original_conf       = original_confidence,
            attacked_conf       = attacked_conf,
            confidence_drop     = conf_drop,
            verdict_flipped     = verdict_flipped,
            status              = status,
            robustness_score    = robustness,
            highlighted_changes = changes,
            technique_details   = attack["technique"],
        ))

    # ── Aggregate ─────────────────────────────────────────────────────────
    summary = {
        "passed":  sum(1 for r in results if r.status == TestStatus.PASSED),
        "warning": sum(1 for r in results if r.status == TestStatus.WARNING),
        "failed":  sum(1 for r in results if r.status == TestStatus.FAILED),
    }

    overall_robustness = round(
        sum(r.robustness_score for r in results) / len(results), 1
    )

    if summary["failed"] > 0:
        overall_status = TestStatus.FAILED
    elif summary["warning"] > 1:
        overall_status = TestStatus.WARNING
    else:
        overall_status = TestStatus.PASSED

    # ── Recommendations ───────────────────────────────────────────────────
    recommendations: list[str] = []
    if summary["failed"] > 0:
        recommendations.append(
            "One or more attacks flipped the verdict. Consider retraining with adversarial examples."
        )
    if any(r.attack_name == "GPT-4 Humanisation Rewrite" and r.status != TestStatus.PASSED for r in results):
        recommendations.append(
            "GPT-4 rewrite reduced confidence significantly. Strengthen perplexity-based features."
        )
    if any(r.attack_type == "paraphrase" and r.confidence_drop > 12 for r in results):
        recommendations.append(
            "Paraphrase attacks caused notable confidence drops — consider adding semantic embedding features."
        )
    if overall_robustness >= 85:
        recommendations.append(
            "Overall robustness is strong. System is resilient to common evasion techniques."
        )

    return AdversarialReport(
        filename            = filename,
        original_verdict    = original_verdict,
        original_confidence = original_confidence,
        overall_robustness  = overall_robustness,
        overall_status      = overall_status,
        results             = results,
        summary             = summary,
        recommendations     = recommendations,
    )