# ============================================================
# FEMCARE — CROSS CHECK: WITH vs WITHOUT OVARIAN CYSTS
# Trains AdaBoost on both feature sets and compares metrics
# ============================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score,
                             f1_score, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# ── 0. LOAD DATA ─────────────────────────────────────────────
df = pd.read_excel("considerable dataset.xlsx")

with open('final_features.txt', 'r') as f:
    lines = f.readlines()

BASE_FEATURES = []
for line in lines:
    line = line.strip()
    if line and line[0].isdigit():
        feature_name = line.split('. ', 1)[1].strip()
        BASE_FEATURES.append(feature_name)

# Current model — without ovarian cysts
FEATURES_WITHOUT = [f for f in BASE_FEATURES if 'Ovarian' not in f]

# With ovarian cysts added back
FEATURES_WITH = FEATURES_WITHOUT + ['Ovarian cysts']

y = df['label']

# ── 1. HELPER FUNCTION ────────────────────────────────────────
def evaluate(features, label, y, name):
    X = df[features]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    # Youden threshold
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred = (y_prob >= best_threshold).astype(int)

    # Metrics
    auc      = roc_auc_score(y_test, y_prob)
    acc      = accuracy_score(y_test, y_pred)
    sens     = recall_score(y_test, y_pred)
    spec     = recall_score(y_test, y_pred, pos_label=0)
    f1       = f1_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # CV-AUC
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc   = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

    print(f"\n{'='*60}")
    print(f"MODEL: {name}")
    print(f"Features used: {len(features)}")
    print(f"{'='*60}")
    print(f"  Test AUC         : {auc:.4f}")
    print(f"  CV-AUC (5-fold)  : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    print(f"  Accuracy         : {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Sensitivity      : {sens:.4f} ({sens*100:.1f}%)")
    print(f"  Specificity      : {spec:.4f} ({spec*100:.1f}%)")
    print(f"  F1 Score         : {f1:.4f}")
    print(f"  Threshold used   : {best_threshold:.4f}")
    print(f"  TP (Endo found)  : {tp}")
    print(f"  FN (Endo missed) : {fn}")
    print(f"  FP (False alarm) : {fp}")
    print(f"  TN (Cleared)     : {tn}")

    return {
        'AUC': auc, 'CV-AUC': cv_auc.mean(),
        'Accuracy': acc, 'Sensitivity': sens,
        'Specificity': spec, 'F1': f1,
        'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn
    }

# ── 2. RUN BOTH MODELS ────────────────────────────────────────
print("Running both models...")

results_without = evaluate(FEATURES_WITHOUT, 'Without Ovarian Cysts', y, 
                           f'WITHOUT Ovarian Cysts ({len(FEATURES_WITHOUT)} features)')
results_with    = evaluate(FEATURES_WITH, 'With Ovarian Cysts', y,
                           f'WITH Ovarian Cysts ({len(FEATURES_WITH)} features)')

# ── 3. SIDE BY SIDE COMPARISON ────────────────────────────────
print(f"\n{'='*60}")
print("SIDE BY SIDE COMPARISON")
print(f"{'='*60}")
print(f"{'Metric':<20} {'Without Cysts':>15} {'With Cysts':>12} {'Difference':>12}")
print("-" * 60)

metrics = ['AUC', 'CV-AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1']
for m in metrics:
    wo  = results_without[m]
    wi  = results_with[m]
    diff = wi - wo
    sign = "+" if diff > 0 else ""
    print(f"  {m:<18} {wo:>15.4f} {wi:>12.4f} {sign+f'{diff:.4f}':>12}")

print()
print(f"  {'Endo missed (FN)':<18} {results_without['FN']:>15} {results_with['FN']:>12} {results_with['FN']-results_without['FN']:>12}")
print(f"  {'False alarms (FP)':<18} {results_without['FP']:>15} {results_with['FP']:>12} {results_with['FP']-results_without['FP']:>12}")
print()
print("="*60)
print("Positive difference = 'With Cysts' is better")
print("Negative difference = 'Without Cysts' is better")
print("="*60)