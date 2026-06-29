# ============================================================
# FEMCARE — SAVE THRESHOLD + PRINT METRICS
# Retrains AdaBoost (identical model, random_state=42),
# calculates Youden threshold, saves it, prints full metrics
# ============================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, roc_auc_score, accuracy_score,
                             recall_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── 0. LOAD DATA ─────────────────────────────────────────────
df = pd.read_excel("considerable dataset.xlsx")

with open('final_features.txt', 'r') as f:
    lines = f.readlines()

FINAL_FEATURES = []
for line in lines:
    line = line.strip()
    if line and line[0].isdigit():
        feature_name = line.split('. ', 1)[1].strip()
        FINAL_FEATURES.append(feature_name)

FINAL_FEATURES = [f for f in FINAL_FEATURES if 'Ovarian' not in f]

X = df[FINAL_FEATURES]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ── 1. RETRAIN (identical to saved model) ────────────────────
ada_model = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
ada_model.fit(X_train, y_train)

# ── 2. CALCULATE YOUDEN THRESHOLD ────────────────────────────
y_prob = ada_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
best_threshold = thresholds[np.argmax(tpr - fpr)]

# ── 3. SAVE THRESHOLD ─────────────────────────────────────────
joblib.dump(best_threshold, 'femcare_threshold.pkl')

# ── 4. PREDICTIONS AT YOUDEN THRESHOLD ───────────────────────
y_pred = (y_prob >= best_threshold).astype(int)

# ── 5. METRICS ────────────────────────────────────────────────
auc         = roc_auc_score(y_test, y_prob)
accuracy    = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)
specificity = recall_score(y_test, y_pred, pos_label=0)
f1          = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("=" * 60)
print("FEMCARE — ADABOOST FINAL MODEL METRICS")
print("=" * 60)
print(f"  Threshold (Youden-optimal) : {best_threshold:.4f}")
print()
print(f"  AUC                        : {auc:.4f}")
print(f"  Accuracy                   : {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"  Sensitivity (Recall)       : {sensitivity:.4f} ({sensitivity*100:.1f}%)")
print(f"  Specificity                : {specificity:.4f} ({specificity*100:.1f}%)")
print(f"  F1 Score                   : {f1:.4f}")
print()
print("  Confusion Matrix:")
print(f"  TP — Endo correctly found  : {tp}")
print(f"  FN — Endo missed           : {fn}")
print(f"  FP — False alarms          : {fp}")
print(f"  TN — Healthy correctly cleared : {tn}")
print()
print(">> Saved: femcare_threshold.pkl")
print("=" * 60)

