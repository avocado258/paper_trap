"""
PaperTrap - ML Training Pipeline
Ensemble: Random Forest + Gradient Boosting (soft voting)
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# ── 1. Load & preprocess ──────────────────────────────────────────────────────

df = pd.read_csv("outputs/features.csv")

# Clip outliers from bad PDF extraction
df['sent_len_mean'] = df['sent_len_mean'].clip(upper=60)
df['word_len_mean'] = df['word_len_mean'].clip(lower=3.0, upper=9.0)

X = df.drop(columns=['label', 'citation_density', 'perplexity', 'sent_len_std'])
y = df['label']

feature_names = X.columns.tolist()
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class distribution:\n{y.value_counts()}")

# ── 2. Train/Val/Test split (70/15/15 stratified) ────────────────────────────

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15 / 0.85, stratify=y_temp, random_state=42
)

print(f"\nSplit sizes — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ── 3. Base estimators (no scaling needed for tree models) ───────────────────

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

gbm = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_samples_split=4,
    min_samples_leaf=2,
    subsample=0.8,
    max_features='sqrt',
    random_state=42
)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gbm', gbm)],
    voting='soft',
    weights=[1, 1]      # equal weight; tune if one dominates on val set
)

# ── 4. Hyperparameter tuning (RF only — GBM already conservatively set) ──────

RF_PARAM_GRID = {
    'n_estimators': [200, 300],
    'max_depth': [None, 20],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 0.5],
}

print("\nRunning GridSearchCV for Random Forest...")
cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_search = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    RF_PARAM_GRID,
    cv=cv_inner,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
rf_search.fit(X_train, y_train)
best_rf_params = rf_search.best_params_
print(f"Best RF params: {best_rf_params}")

# Rebuild with tuned RF
rf_tuned = RandomForestClassifier(
    **best_rf_params,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

ensemble_tuned = VotingClassifier(
    estimators=[('rf', rf_tuned), ('gbm', gbm)],
    voting='soft'
)

# ── 5. 5-fold stratified cross-validation (on full train set) ────────────────

print("\nRunning 5-fold CV on ensemble...")
cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    ensemble_tuned,
    X_train, y_train,
    cv=cv_outer,
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    return_train_score=False,
    n_jobs=-1
)

print("\nCross-Validation Results (mean ± std):")
for metric, scores in cv_results.items():
    if metric.startswith('test_'):
        name = metric.replace('test_', '')
        print(f"  {name:12s}: {scores.mean():.4f} ± {scores.std():.4f}")

# ── 6. Final training on train split, evaluate on val ────────────────────────

ensemble_tuned.fit(X_train, y_train)

def evaluate(model, X, y, split_name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    metrics = {
        'accuracy':  accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall':    recall_score(y, y_pred),
        'f1':        f1_score(y, y_pred),
        'roc_auc':   roc_auc_score(y, y_prob),
        'confusion_matrix': confusion_matrix(y, y_pred).tolist()
    }
    print(f"\n── {split_name} ──")
    for k, v in metrics.items():
        if k != 'confusion_matrix':
            print(f"  {k:12s}: {v:.4f}")
    print(f"  Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}")
    print(classification_report(y, y_pred, target_names=['Real', 'Fake']))
    return metrics

val_metrics  = evaluate(ensemble_tuned, X_val,  y_val,  "Validation Set")
test_metrics = evaluate(ensemble_tuned, X_test, y_test, "Test Set")

# ── 7. Feature importance (RF component) ─────────────────────────────────────

rf_fitted = ensemble_tuned.named_estimators_['rf']
importances = pd.Series(rf_fitted.feature_importances_, index=feature_names)
importances_sorted = importances.sort_values(ascending=False)

print("\nTop 20 Feature Importances (RF):")
print(importances_sorted.head(20).to_string())

# ── 8. Save artifacts ─────────────────────────────────────────────────────────

Path("outputs/models").mkdir(parents=True, exist_ok=True)

joblib.dump(ensemble_tuned, "outputs/models/ensemble.pkl")
joblib.dump(rf_fitted,      "outputs/models/rf_component.pkl")
importances_sorted.to_csv("outputs/models/feature_importances.csv", header=['importance'])

results = {
    'cv': {k.replace('test_', ''): {'mean': float(v.mean()), 'std': float(v.std())}
           for k, v in cv_results.items() if k.startswith('test_')},
    'val':  {k: v for k, v in val_metrics.items()  if k != 'confusion_matrix'},
    'test': {k: v for k, v in test_metrics.items() if k != 'confusion_matrix'},
    'best_rf_params': best_rf_params,
    'feature_names': feature_names
}

with open("outputs/models/results.json", 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Saved: outputs/models/ensemble.pkl, rf_component.pkl, feature_importances.csv, results.json")