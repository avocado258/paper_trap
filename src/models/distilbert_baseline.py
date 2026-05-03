"""
PaperTrap — Baseline 1: Fine-tuned DistilBERT
- Classification head + last 2 transformer layers unfrozen
- 5-fold stratified cross-validation with std reporting
- Returns per-fold metrics + aggregate report
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME    = "distilbert-base-uncased"
MAX_LENGTH    = 512          # DistilBERT hard limit
BATCH_SIZE    = 8
EPOCHS        = 4
LR            = 2e-5
WARMUP_RATIO  = 0.1
N_FOLDS       = 5
SEED          = 42
MAX_WORDS     = 3000         # truncate raw text before tokenization (project spec)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Helpers ───────────────────────────────────────────────────────────────────

def truncate_to_words(text: str, n: int = MAX_WORDS) -> str:
    return " ".join(text.split()[:n])


class PaperDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }


def build_model():
    """Fresh DistilBERT with only classification head + last 2 layers unfrozen."""
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    # Freeze all layers first
    for param in model.distilbert.parameters():
        param.requires_grad = False

    # Unfreeze last 2 transformer blocks (layers 4 and 5 in 6-layer DistilBERT)
    for block in model.distilbert.transformer.layer[-2:]:
        for param in block.parameters():
            param.requires_grad = True

    # Classification head is always trainable (default)
    return model.to(DEVICE)


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds   = np.argmax(probs, axis=1)

        all_probs.extend(probs[:, 1])   # prob of class 1 (fake)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ── Main CV Loop ──────────────────────────────────────────────────────────────

def run_distilbert_cv(texts: list[str], labels: list[int]):
    """
    Parameters
    ----------
    texts  : raw paper text (will be word-truncated internally)
    labels : 0 = human, 1 = AI-generated

    Returns
    -------
    fold_results : list of per-fold metric dicts
    summary      : dict with mean ± std for key metrics
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    texts_trunc = [truncate_to_words(t) for t in texts]

    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    X    = np.array(texts_trunc)
    y    = np.array(labels)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'='*50}")
        print(f"Fold {fold}/{N_FOLDS}")
        print(f"{'='*50}")

        train_texts, val_texts   = X[train_idx].tolist(), X[val_idx].tolist()
        train_labels, val_labels = y[train_idx].tolist(), y[val_idx].tolist()

        train_ds = PaperDataset(train_texts, train_labels, tokenizer)
        val_ds   = PaperDataset(val_texts,   val_labels,   tokenizer)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

        model     = build_model()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR, weight_decay=0.01
        )
        total_steps   = len(train_loader) * EPOCHS
        warmup_steps  = int(total_steps * WARMUP_RATIO)
        scheduler     = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        best_val_f1, best_state = 0.0, None

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, scheduler)
            val_labels_arr, val_preds, val_probs = evaluate(model, val_loader)
            val_f1 = f1_score(val_labels_arr, val_preds)
            print(f"  Epoch {epoch} | loss={train_loss:.4f} | val_f1={val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Reload best checkpoint for final eval
        model.load_state_dict(best_state)
        model.to(DEVICE)
        true, preds, probs = evaluate(model, val_loader)

        metrics = {
            "fold":      fold,
            "accuracy":  accuracy_score(true, preds),
            "precision": precision_score(true, preds, zero_division=0),
            "recall":    recall_score(true, preds, zero_division=0),
            "f1":        f1_score(true, preds, zero_division=0),
            "roc_auc":   roc_auc_score(true, probs),
        }
        fold_results.append(metrics)
        print(f"\n  Fold {fold} results: {metrics}")

        # Free GPU memory between folds
        del model, optimizer, scheduler, train_ds, val_ds
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Aggregate ──────────────────────────────────────────────────────────────
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    summary = {}
    print(f"\n{'='*50}")
    print("DistilBERT Cross-Validation Summary")
    print(f"{'='*50}")
    for k in metric_keys:
        vals = [r[k] for r in fold_results]
        summary[k] = {"mean": np.mean(vals), "std": np.std(vals)}
        print(f"  {k:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    return fold_results, summary


def final_distilbert_eval(
    train_texts, train_labels,
    test_texts,  test_labels
):
    """
    Train on full train split, evaluate on held-out test set.
    Call after CV to get the final confusion matrix / classification report.
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_texts = [truncate_to_words(t) for t in train_texts]
    test_texts  = [truncate_to_words(t) for t in test_texts]

    train_ds    = PaperDataset(train_texts, train_labels, tokenizer)
    test_ds     = PaperDataset(test_texts,  test_labels,  tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    model     = build_model()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=0.01
    )
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch} | loss={loss:.4f}")

    true, preds, probs = evaluate(model, test_loader)

    print("\n=== DistilBERT — Test Set Results ===")
    print(classification_report(true, preds, target_names=["Human", "AI-Generated"]))
    print(f"ROC-AUC: {roc_auc_score(true, probs):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(true, preds)}")

    return model, true, preds, probs


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import os
    from sklearn.model_selection import train_test_split as sk_split

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv("data/final_dataset.csv")
    df["label_int"] = (df["label"] == "AI").astype(int)   # Human=0, AI=1

    texts  = df["text"].tolist()
    labels = df["label_int"].tolist()

    print(f"Dataset: {len(texts)} samples")
    print(f"Class distribution:\n{df['label'].value_counts()}")

    # ── Replicate identical 70/15/15 split from ensemble training ────────────
    X_temp, X_test, y_temp, y_test = sk_split(
        texts, labels, test_size=0.15, stratify=labels, random_state=42
    )
    X_train, X_val, y_train, y_val = sk_split(
        X_temp, y_temp, test_size=0.15 / 0.85, stratify=y_temp, random_state=42
    )
    print(f"Split — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ── 5-fold CV on train split ──────────────────────────────────────────────
    fold_results, cv_summary = run_distilbert_cv(X_train, y_train)

    # ── Final evaluation on held-out test set ─────────────────────────────────
    print("\nTraining final model on full train split for test-set evaluation...")
    model, true, preds, probs = final_distilbert_eval(
        X_train, y_train, X_test, y_test
    )

    # ── Save CV summary (consumed by compare_baselines in zero_shot_baseline) ─
    os.makedirs("outputs/models", exist_ok=True)
    with open("outputs/models/distilbert_cv_summary.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    print("\n✓ Saved: outputs/models/distilbert_cv_summary.json")