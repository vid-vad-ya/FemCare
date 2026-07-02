"""
femcare_roc_curve.py
---------------------
Generates ROC curve for the production femcare_model_final.pkl
Run from D:\FemCare\

Output: roc_curve_final.png
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

# ── Load model and features ──────────────────────────────────
model = joblib.load("femcare_model_final.pkl")

with open("femcare_features.txt", "r") as f:
    FEATURES = [line.strip() for line in f if line.strip()]

# ── Load dataset ─────────────────────────────────────────────
df = pd.read_excel("considerable dataset.xlsx")
X  = df[FEATURES]
y  = df["label"]

# ── Same exact split as training ─────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ── Get probabilities ─────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]

# ── Compute ROC ──────────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc              = auc(fpr, tpr)

# ── Youden optimal point ─────────────────────────────────────
youden_idx       = np.argmax(tpr - fpr)
optimal_thresh   = thresholds[youden_idx]
optimal_fpr      = fpr[youden_idx]
optimal_tpr      = tpr[youden_idx]

# ── CV AUC ───────────────────────────────────────────────────
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

print("=" * 50)
print("  ROC CURVE METRICS")
print("=" * 50)
print(f"  Test AUC              : {roc_auc:.4f}")
print(f"  CV-AUC (5-fold mean)  : {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")
print(f"  Optimal threshold     : {optimal_thresh:.4f}")
print(f"  At optimal point:")
print(f"    Sensitivity (TPR)   : {optimal_tpr:.4f} ({optimal_tpr*100:.1f}%)")
print(f"    1 - Specificity (FPR): {optimal_fpr:.4f} ({optimal_fpr*100:.1f}%)")
print(f"    Specificity         : {(1-optimal_fpr):.4f} ({(1-optimal_fpr)*100:.1f}%)")
print("=" * 50)

# ── Plot ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))

# Diagonal reference line
ax.plot([0, 1], [0, 1],
        color="#cccccc", linestyle="--", linewidth=1.5,
        label="Random Classifier (AUC = 0.50)")

# ROC curve with gradient fill
ax.plot(fpr, tpr,
        color="#7c3aed", linewidth=2.5,
        label=f"AdaBoost Calibrated (AUC = {roc_auc:.4f})")

# Fill under curve
ax.fill_between(fpr, tpr, alpha=0.08, color="#7c3aed")

# CV AUC band annotation
ax.annotate(
    f"CV-AUC = {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}\n(5-fold stratified)",
    xy=(0.55, 0.18),
    fontsize=10, color="#7c3aed",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f0ff", edgecolor="#c4b5fd")
)

# Optimal threshold point
ax.scatter([optimal_fpr], [optimal_tpr],
           color="#e11d48", s=120, zorder=5,
           label=f"Optimal threshold = {optimal_thresh:.4f}\n"
                 f"(Sensitivity={optimal_tpr*100:.1f}%, Specificity={(1-optimal_fpr)*100:.1f}%)")

# Dotted lines to axes from optimal point
ax.plot([optimal_fpr, optimal_fpr], [0, optimal_tpr],
        color="#e11d48", linestyle=":", linewidth=1, alpha=0.6)
ax.plot([0, optimal_fpr], [optimal_tpr, optimal_tpr],
        color="#e11d48", linestyle=":", linewidth=1, alpha=0.6)

# Labels and formatting
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
ax.set_title("ROC Curve — FemCare Endometriosis Risk Model\n"
             "Calibrated AdaBoost · 886 patients · 14 features",
             fontsize=13, fontweight="bold", pad=15)

ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
ax.grid(alpha=0.25, linestyle="--")
ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)

# Metrics box
metrics_text = (
    f"Test AUC     : {roc_auc:.4f}\n"
    f"CV-AUC       : {cv_aucs.mean():.4f}\n"
    f"Sensitivity  : {optimal_tpr*100:.1f}%\n"
    f"Specificity  : {(1-optimal_fpr)*100:.1f}%\n"
    f"Threshold    : {optimal_thresh:.4f}"
)
ax.text(0.63, 0.52, metrics_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5",
                  facecolor="white", edgecolor="#ddd6fe", alpha=0.95))

plt.tight_layout()
plt.savefig("roc_curve_final.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n>> Saved: roc_curve_final.png")