"""
PaperTrap - AI-Generated Research Paper Generator
File: papertrap/src/data_collection/paper_generator.py

Usage:
    python paper_generator.py --member 1   # generates papers 1–84   (Gemini)
    python paper_generator.py --member 2   # generates papers 85–168  (Gemini)
    python paper_generator.py --member 3   # generates papers 169–250 (Gemini)

Each member only needs their own Gemini API key.
Run the script, it saves .txt files to papertrap/data/fake_papers/
and appends rows to papertrap/data/fake_papers_metadata.csv.
Crash-safe: re-running resumes from where it left off.
"""

import os
import csv
import json
import time
import random
import logging
import argparse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# pip install google-generativeai
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (relative to this file's location: src/data_collection/)
# ─────────────────────────────────────────────────────────────────────────────

THIS_DIR      = Path(__file__).resolve().parent          # src/data_collection/
PROJECT_ROOT  = THIS_DIR.parent.parent                   # papertrap/
DATA_DIR      = PROJECT_ROOT / "data"
FAKE_DIR      = DATA_DIR / "fake_papers"
METADATA_CSV  = DATA_DIR / "fake_papers_metadata.csv"
PROGRESS_FILE = DATA_DIR / "fake_papers_progress.json"
LOG_FILE      = DATA_DIR / "fake_papers_generation.log"


# ─────────────────────────────────────────────────────────────────────────────
# API KEYS — set via environment variables or fill in directly
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")


# ─────────────────────────────────────────────────────────────────────────────
# MEMBER ASSIGNMENTS — all Gemini
# Member 1: papers   1–84
# Member 2: papers  85–168
# Member 3: papers 169–250
# Each member only needs their own Gemini API key.
# ─────────────────────────────────────────────────────────────────────────────

MEMBER_CONFIG = {
    1: {"start": 1,   "end": 84,  "providers": ["gemini"]},
    2: {"start": 85,  "end": 168, "providers": ["gemini"]},
    3: {"start": 169, "end": 250, "providers": ["gemini"]},
}

TOTAL_PAPERS   = 250
RETRY_ATTEMPTS = 3
RETRY_DELAY    = 10   # base seconds (multiplied by attempt number)
REQUEST_DELAY  = 4    # seconds between successful requests


# ─────────────────────────────────────────────────────────────────────────────
# TOPICS
# ─────────────────────────────────────────────────────────────────────────────

TOPICS = [
    ("Federated Learning / Medical Image Analysis",
     "Federated Learning for Privacy-Preserving Medical Image Segmentation"),
    ("Federated Learning / Differential Privacy",
     "Differentially Private Gradient Aggregation in Cross-Silo Federated Learning"),
    ("Privacy-Preserving Machine Learning",
     "Secure Multi-Party Computation for Distributed Model Training"),
    ("Graph Neural Networks / Anomaly Detection",
     "Graph Neural Networks for Supply Chain Anomaly Detection"),
    ("Graph Neural Networks / Natural Language Processing",
     "Heterogeneous Graph Transformers for Biomedical Relation Extraction"),
    ("Graph Neural Networks / Recommendation Systems",
     "Dynamic Graph Attention Networks for Session-Based Recommendation"),
    ("Self-Supervised Learning / Speech Processing",
     "Contrastive Self-Supervised Learning for Low-Resource Speech Recognition"),
    ("Self-Supervised Learning / Computer Vision",
     "Masked Autoencoders with Hierarchical Token Prediction for Dense Prediction"),
    ("Contrastive Learning / Multi-Modal",
     "Cross-Modal Contrastive Alignment for Vision-Language Pretraining"),
    ("Natural Language Processing / Efficient Transformers",
     "Sparse Mixture-of-Experts Attention for Long-Document Summarization"),
    ("Natural Language Processing / Transformers",
     "Improving Transformer Efficiency Using Dynamic Token Pruning"),
    ("Natural Language Processing / Low-Resource NLP",
     "Cross-Lingual Transfer Learning for Low-Resource Named Entity Recognition"),
    ("Reinforcement Learning / Robotics",
     "Hierarchical Reinforcement Learning for Multi-Step Manipulation Tasks"),
    ("Reinforcement Learning / Multi-Agent Systems",
     "Cooperative Multi-Agent Reinforcement Learning with Communication Constraints"),
    ("Offline Reinforcement Learning",
     "Conservative Policy Optimization for Offline Reinforcement Learning"),
    ("Explainable AI / Computer Vision",
     "Gradient-Based Saliency Maps with Causal Attribution for Medical Diagnosis"),
    ("Fairness in Machine Learning",
     "Mitigating Demographic Bias in Large Language Models via Contrastive Fine-Tuning"),
    ("Explainable AI / NLP",
     "Attention Flow Analysis for Interpretable Text Classification"),
    ("Generative Models / Diffusion",
     "Latent Diffusion Models for High-Fidelity Medical Image Synthesis"),
    ("Generative Models / GANs",
     "Conditional Wasserstein GANs for Tabular Data Augmentation"),
    ("Large Language Models",
     "Parameter-Efficient Fine-Tuning of Large Language Models via Adaptive LoRA"),
    ("Computer Vision / Object Detection",
     "Anchor-Free Object Detection with Deformable Convolutional Feature Pyramids"),
    ("Computer Vision / Semantic Segmentation",
     "Boundary-Aware Transformer for Weakly Supervised Semantic Segmentation"),
    ("Computer Vision / Few-Shot Learning",
     "Prototype Calibration Networks for Few-Shot Image Classification"),
    ("Time Series Analysis / Anomaly Detection",
     "Temporal Convolutional Networks with Attention for Multivariate Anomaly Detection"),
    ("Time Series Analysis / Forecasting",
     "Probabilistic Transformer Forecasting with Conformal Prediction Intervals"),
    ("Tabular Machine Learning",
     "TabNet Revisited: Gated Feature Selection with Self-Supervised Pretraining"),
    ("Knowledge Graph Embedding",
     "Relational Hyperbolic Embeddings for Knowledge Graph Completion"),
    ("Neural Reasoning",
     "Chain-of-Thought Distillation for Multi-Hop Question Answering"),
    ("Commonsense Reasoning",
     "Neurosymbolic Integration for Commonsense Inference in Open-Domain QA"),
    ("NLP / Clinical Text Mining",
     "Transformer-Based Extraction of Adverse Drug Events from Clinical Notes"),
    ("Computer Vision / Remote Sensing",
     "Self-Supervised Change Detection in Multitemporal Satellite Imagery"),
    ("NLP / Code Generation",
     "Execution-Guided Neural Program Synthesis with Type-Constrained Decoding"),
    ("NLP / Dialogue Systems",
     "Persona-Consistent Response Generation with Memory-Augmented Transformers"),
    ("Computer Vision / 3D Point Clouds",
     "PointFlow: Normalizing Flows for 3D Point Cloud Generation"),
    ("Optimization / Deep Learning",
     "Adaptive Gradient Clipping for Training Stability in Large-Scale Neural Networks"),
    ("Meta-Learning",
     "Task-Agnostic Meta-Learning via Latent Space Optimization"),
    ("Neural Architecture Search",
     "Differentiable Architecture Search with Hardware-Aware Efficiency Constraints"),
    ("Continual Learning",
     "Elastic Weight Consolidation with Dynamic Memory Replay for Continual Learning"),
    ("Multimodal Learning / Vision-Language",
     "Grounded Visual Question Answering with Scene Graph Supervision"),
    ("Multimodal Learning / Audio-Visual",
     "Audio-Visual Source Separation via Cross-Modal Attention Fusion"),
    ("Multimodal Learning / Document Understanding",
     "Layout-Aware Multimodal Transformers for Document Information Extraction"),
    ("Adversarial Robustness",
     "Certified Adversarial Robustness via Randomized Smoothing with Tight Bounds"),
    ("Distribution Shift",
     "Test-Time Adaptation via Self-Supervised Auxiliary Tasks under Covariate Shift"),
    ("Bayesian Deep Learning",
     "Scalable Bayesian Neural Networks via Structured Variational Inference"),
    ("Causal Inference / Machine Learning",
     "Causal Representation Learning for Out-of-Distribution Generalization"),
    ("Neural Compression",
     "Learned Image Compression with Entropy Coding and Hyperprior Models"),
    ("Quantum Machine Learning",
     "Variational Quantum Circuits for Binary Classification on NISQ Devices"),
    ("Mixture of Experts",
     "Sparse Gating Mechanisms for Conditional Computation in Vision Transformers"),
    ("Semi-Supervised Learning",
     "Consistency Regularization with Stochastic Augmentation Graphs for SSL"),
]


# ─────────────────────────────────────────────────────────────────────────────
# STYLOMETRIC DIVERSITY PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

STYLE_VARIANTS = [
    "Write with a concise, direct style — prefer short sentences in methodology. "
    "Use active voice more than passive in the introduction.",

    "Write with an elaborate, discursive style — use longer, multi-clause sentences "
    "and extensive hedging language (e.g., 'it is worth noting', 'arguably', 'one may observe').",

    "Write in a terse, equation-heavy style — minimize prose in methodology, "
    "let equations carry the argument. Use passive voice throughout.",

    "Write with a narrative, motivation-driven style — open each section by "
    "framing the problem before presenting the solution.",

    "Write with a rigorous, theorem-proof style — emphasize formal assumptions, "
    "lemmas, and structured proofs. Minimize informal commentary.",

    "Write with a systems-engineering style — emphasize implementation details, "
    "runtime complexity, and practical trade-offs over theoretical elegance.",
]

CITATION_STYLES = [
    "Use citations aggressively — cite 2–3 references per claim throughout the paper.",
    "Use citations sparingly but precisely — cite only the single most relevant reference per claim.",
    "Cite heavily in Related Work and Introduction, but minimally in Methodology.",
]

STRUCTURE_VARIANTS = [
    "Number all section headings (1. Introduction, 2. Related Work, etc.).",
    "Use unnumbered section headings in the style of ACL papers.",
    "Use Roman numeral section headings (I. Introduction, II. Related Work, etc.).",
]


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────────────────────

BASE_PROMPT = """You are an expert academic researcher and scientific writer in the fields of Artificial Intelligence, Machine Learning, and Natural Language Processing. Your task is to generate a fully detailed, realistic, and technically sophisticated research paper that closely mimics the style, structure, and rigor of top-tier conference/journal papers (e.g., arXiv, NeurIPS, ICML, ACL).

The generated paper must match the following requirements:

--------------------------------------------------
STRUCTURE & FORMAT
--------------------------------------------------
Follow a formal academic structure similar to real research papers:

1. Title (concise, technical, novel-sounding)
2. Authors (realistic fictional names + affiliations)
3. Abstract (150-250 words, dense and technical)
4. Keywords (5-8 relevant terms)
5. Introduction
6. Related Work
7. Background / Preliminaries (include formal definitions if needed)
8. Methodology / Proposed Approach
9. Theoretical Analysis (include equations, assumptions, or proofs where appropriate)
10. Experiments
    - Datasets
    - Experimental Setup
    - Baselines
    - Evaluation Metrics
    - Results (tables, comparisons, discussion)
11. Ablation Study (analyze the contribution of each component of your method)
12. Discussion / Insights
13. Limitations & Future Work
14. Conclusion
15. References (fictional but realistic citations following IEEE format exactly)

--------------------------------------------------
STYLE & TONE
--------------------------------------------------
- Use formal academic language with precise terminology.
- Write in a confident, objective tone.
- Avoid conversational phrasing.
- Include citations in brackets (e.g., [1], [2]) throughout the text.
- Mimic complexity similar to real research papers (dense paragraphs, layered arguments).
- Use passive voice where appropriate.
- Vary writing style across sections:
  - Introduction & Discussion: more elaborate and discursive
  - Methodology & Theory: terse, precise, and technical

--------------------------------------------------
TECHNICAL DEPTH
--------------------------------------------------
- Include mathematical formulations:
  - Define variables, functions, and equations.
  - Use notation consistent with ML/NLP literature.
  - Include at least 4 numbered equations and reference them in the text.
- Introduce a novel-sounding method, algorithm, or framework.
- Include at least one algorithm in pseudocode format describing the proposed method.
- Discuss computational complexity (e.g., O(n log n), etc.).
- Include trade-offs (bias-variance, accuracy vs efficiency, etc.).
- Add theoretical claims (with justification, formal or informal).

--------------------------------------------------
EXPERIMENTAL REALISM
--------------------------------------------------
- Use realistic dataset names (e.g., CIFAR-10, ImageNet, GLUE, custom datasets).
- Provide plausible numerical results (do NOT make them perfect).
- Compare against at least 5 baseline models.
- Include:
  - At least TWO clearly formatted results tables with labeled rows, columns, and metrics
  - One ablation study table breaking down component contributions
  - Observations about performance trends
- Mention hyperparameters, training setup, and evaluation metrics.

--------------------------------------------------
DOMAIN CONTROL
--------------------------------------------------
The paper should be in the domain of: {domain}
The specific topic should be: {topic}

If no domain or topic is provided, automatically select a specific, narrow, and technically interesting research problem within AI/ML/NLP.

--------------------------------------------------
INNOVATION REQUIREMENT
--------------------------------------------------
The paper must propose something that sounds novel, such as:
- A new algorithm or architecture
- A modification of an existing method
- A hybrid approach combining multiple techniques
- A new theoretical insight or framework

--------------------------------------------------
REALISM CONSTRAINTS
--------------------------------------------------
- Do NOT mention that the paper is fake or generated.
- Avoid exaggerated claims (e.g., "100% accuracy").
- Include limitations and trade-offs.
- Ensure consistency across sections (method -> experiments -> results).

--------------------------------------------------
CITATION CONSISTENCY
--------------------------------------------------
- Every in-text citation (e.g., [1], [2]) must have a corresponding entry in the References section.
- Every entry in the References section must be cited at least once in the body of the paper.
- No orphaned references and no uncited in-text markers.
- References must follow IEEE citation format exactly:
  [N] A. Lastname, B. Lastname, "Title of paper," in Proc. Conference Name, City, Year, pp. XXX-XXX.
- Reference numbers must be sequential and consistent throughout the paper.
- The paper MUST include at least 25 references.

--------------------------------------------------
LENGTH & COMPLETENESS
--------------------------------------------------
- The paper MUST be at least 4000 words. Do not truncate any section.
- Every section must be fully developed — no section should be less than 3 substantial paragraphs.
- The Discussion, Limitations, and Conclusion sections must each be at least 3 paragraphs.
- Do not summarize or shorten any section due to length — write each section to its full natural depth.

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------
Return the full paper as a structured text document with clear section headings,
formatted for easy conversion into PDF (LaTeX-style or clean academic Markdown).

--------------------------------------------------
STYLE INSTRUCTION
--------------------------------------------------
{style_variant}

--------------------------------------------------
CITATION DENSITY
--------------------------------------------------
{citation_style}

--------------------------------------------------
SECTION HEADING FORMAT
--------------------------------------------------
{structure_variant}
"""


def build_prompt(domain: str, topic: str) -> str:
    return BASE_PROMPT.format(
        domain=domain,
        topic=topic,
        style_variant=random.choice(STYLE_VARIANTS),
        citation_style=random.choice(CITATION_STYLES),
        structure_variant=random.choice(STRUCTURE_VARIANTS),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(),
        ],
    )

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PROGRESS & FILE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_existing_ids() -> set:
    """Read already-saved paper IDs from the fake_papers directory."""
    if not FAKE_DIR.exists():
        return set()
    return {
        int(p.stem.replace("fake_paper_", ""))
        for p in FAKE_DIR.glob("fake_paper_*.txt")
    }


def init_metadata_csv():
    """Create metadata CSV with headers if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not METADATA_CSV.exists():
        with open(METADATA_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "id", "filename", "label", "provider",
                "domain", "topic", "word_count", "timestamp"
            ])
            writer.writeheader()
        logger.info(f"Created {METADATA_CSV}")


def append_metadata(row: dict):
    with open(METADATA_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "filename", "label", "provider",
            "domain", "topic", "word_count", "timestamp"
        ])
        writer.writerow(row)


def save_paper_txt(paper_id: int, text: str) -> Path:
    FAKE_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"fake_paper_{paper_id:03d}.txt"
    filepath = FAKE_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    return filepath


# ─────────────────────────────────────────────────────────────────────────────
# API CALLS
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(prompt: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            max_output_tokens=8192,
            temperature=round(random.uniform(0.7, 1.0), 2),
        ),
    )
    return response.text


def generate_with_retry(provider: str, prompt: str) -> str | None:
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return call_gemini(prompt)
        except Exception as e:
            logger.warning(f"Attempt {attempt}/{RETRY_ATTEMPTS} failed [gemini]: {e}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY * attempt)
    logger.error("All retries exhausted for gemini")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# JOB LIST  (deterministic — same seed = same assignments every run)
# ─────────────────────────────────────────────────────────────────────────────

def build_all_jobs() -> list[dict]:
    """
    Build the full 250-paper job list with deterministic assignments.
    Member 1 : IDs   1–84  → gemini
    Member 2 : IDs  85–168 → gemini
    Member 3 : IDs 169–209 → openai  (41 papers)
               IDs 210–250 → anthropic (41 papers)
    Topics are cycled deterministically across IDs.
    """
    rng = random.Random(42)  # fixed seed — reproducible assignments
    topics_shuffled = TOPICS.copy()
    rng.shuffle(topics_shuffled)

    jobs = []
    for paper_id in range(1, TOTAL_PAPERS + 1):
        domain, topic = topics_shuffled[(paper_id - 1) % len(topics_shuffled)]

        provider = "gemini"

        jobs.append({
            "id":       paper_id,
            "provider": provider,
            "domain":   domain,
            "topic":    topic,
        })
    return jobs


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(member: int):
    setup_logging()
    logger.info(f"=== PaperTrap Generator | Member {member} ===")

    config = MEMBER_CONFIG[member]
    start, end = config["start"], config["end"]
    logger.info(f"Assigned range: papers {start}–{end} ({end - start + 1} papers)")

    init_metadata_csv()
    FAKE_DIR.mkdir(parents=True, exist_ok=True)

    all_jobs  = build_all_jobs()
    my_jobs   = [j for j in all_jobs if start <= j["id"] <= end]

    existing  = get_existing_ids()
    pending   = [j for j in my_jobs if j["id"] not in existing]

    logger.info(f"Already done : {len(existing) & len({j['id'] for j in my_jobs})} / {len(my_jobs)}")
    logger.info(f"Pending      : {len(pending)}")

    success = 0
    failed  = 0

    try:
        for job in pending:
            paper_id = job["id"]
            provider = job["provider"]
            domain   = job["domain"]
            topic    = job["topic"]

            logger.info(f"[{paper_id}/{TOTAL_PAPERS}] {provider} | {topic[:60]}")

            prompt = build_prompt(domain, topic)
            text   = generate_with_retry(provider, prompt)

            if text is None:
                logger.error(f"Skipping paper {paper_id} — all retries failed.")
                failed += 1
                continue

            # Save .txt file
            filepath = save_paper_txt(paper_id, text)

            # Append metadata row
            append_metadata({
                "id":         paper_id,
                "filename":   filepath.name,
                "label":      "AI",
                "provider":   provider,
                "domain":     domain,
                "topic":      topic,
                "word_count": len(text.split()),
                "timestamp":  datetime.utcnow().isoformat(),
            })

            success += 1
            logger.info(f"  Saved {filepath.name} ({len(text.split())} words)")
            time.sleep(REQUEST_DELAY)

    except KeyboardInterrupt:
        logger.warning("Interrupted — progress saved.")

    finally:
        done = len(get_existing_ids() & {j["id"] for j in my_jobs})
        logger.info("=== Session Summary ===")
        logger.info(f"  Generated : {success}")
        logger.info(f"  Failed    : {failed}")
        logger.info(f"  Total done (my range): {done} / {len(my_jobs)}")
        if done < len(my_jobs):
            logger.info("  Re-run the script to continue from where it left off.")
        else:
            logger.info("  All papers in your range generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaperTrap fake paper generator")
    parser.add_argument(
        "--member", type=int, required=True, choices=[1, 2, 3],
        help="Group member number (1, 2, or 3). Determines which papers to generate."
    )
    args = parser.parse_args()
    main(args.member)