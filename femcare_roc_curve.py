"""
femcare_plots_final.py
-----------------------
Generates 3 publication-quality plots for FemCare results section:
  1. ROC Curve
  2. Confusion Matrix
  3. Radar Chart (model performance across metrics)

Run from D:\FemCare\
Output: roc_curve_final.png, confusion_matrix_final.png, radar_final.png
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                             accuracy_score, recall_score, f1_score,
                             roc_auc_score)
import math

# ── Load model and features ──────────────────────────────────
print("Loading model and data...")
model = joblib.load("femcare_model_final.pkl")

with open("femcare_features.txt", "r") as f:
    FEATURES = [line.strip() for line in f if line.strip()]

df = pd.read_excel("considerable dataset.xlsx")
X  = df[FEATURES]
y  = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc_val = auc(fpr, tpr)

youden_idx     = np.argmax(tpr - fpr)
optimal_thresh = thresholds[youden_idx]
optimal_fpr    = fpr[youden_idx]
optimal_tpr    = tpr[youden_idx]

y_pred = (y_prob >= optimal_thresh).astype(int)

cm             = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

acc  = accuracy_score(y_test, y_pred)
sens = recall_score(y_test, y_pred)
spec = recall_score(y_test, y_pred, pos_label=0)
f1   = f1_score(y_test, y_pred)

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
cv_auc  = cv_aucs.mean()

print("=" * 55)
print("  FINAL MODEL METRICS")
print("=" * 55)
print(f"  Test AUC         : {roc_auc_val:.4f}")
print(f"  CV-AUC (5-fold)  : {cv_auc:.4f} ± {cv_aucs.std():.4f}")
print(f"  Accuracy         : {acc*100:.2f}%")
print(f"  Sensitivity      : {sens*100:.2f}%")
print(f"  Specificity      : {spec*100:.2f}%")
print(f"  F1 Score         : {f1:.4f}")
print(f"  Threshold        : {optimal_thresh:.4f}")
print(f"  TP={tp}  FN={fn}  FP={fp}  TN={tn}")
print("=" * 55)


# ════════════════════════════════════════════════════════════
# PLOT 1 — ROC CURVE
# ════════════════════════════════════════════════════════════
print("\nGenerating ROC curve...")

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor("white")

ax.plot([0, 1], [0, 1],
        color="#cccccc", linestyle="--", linewidth=1.5,
        label="Random Classifier (AUC = 0.50)")

ax.plot(fpr, tpr,
        color="#7c3aed", linewidth=2.5,
        label=f"Calibrated AdaBoost (AUC = {roc_auc_val:.4f})")

ax.fill_between(fpr, tpr, alpha=0.08, color="#7c3aed")

ax.scatter([optimal_fpr], [optimal_tpr],
           color="#e11d48", s=140, zorder=5,
           label=f"Optimal threshold = {optimal_thresh:.4f}\n"
                 f"Sensitivity = {optimal_tpr*100:.1f}%  "
                 f"Specificity = {(1-optimal_fpr)*100:.1f}%")

ax.plot([optimal_fpr, optimal_fpr], [0, optimal_tpr],
        color="#e11d48", linestyle=":", linewidth=1.2, alpha=0.6)
ax.plot([0, optimal_fpr], [optimal_tpr, optimal_tpr],
        color="#e11d48", linestyle=":", linewidth=1.2, alpha=0.6)

ax.annotate(
    f"CV-AUC = {cv_auc:.4f} ± {cv_aucs.std():.4f}\n(5-fold stratified)",
    xy=(0.52, 0.16), fontsize=10, color="#7c3aed",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f0ff", edgecolor="#c4b5fd")
)

metrics_text = (
    f"Test AUC     : {roc_auc_val:.4f}\n"
    f"CV-AUC       : {cv_auc:.4f}\n"
    f"Sensitivity  : {sens*100:.1f}%\n"
    f"Specificity  : {spec*100:.1f}%\n"
    f"Accuracy     : {acc*100:.1f}%\n"
    f"F1 Score     : {f1:.4f}"
)
ax.text(0.62, 0.50, metrics_text,
        transform=ax.transAxes, fontsize=9.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5",
                  facecolor="white", edgecolor="#ddd6fe", alpha=0.95))

ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
ax.set_title("ROC Curve — FemCare Endometriosis Risk Model\n"
             "Calibrated AdaBoost · 886 patients · 14 features",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
ax.grid(alpha=0.25, linestyle="--")
ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)

plt.tight_layout()
plt.savefig("roc_curve_final.png", dpi=150, bbox_inches="tight")
plt.close()
print(">> Saved: roc_curve_final.png")


# ════════════════════════════════════════════════════════════
# PLOT 2 — CONFUSION MATRIX
# ════════════════════════════════════════════════════════════
print("Generating confusion matrix...")

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor("white")

values    = np.array([[tp, fn], [fp, tn]])
labels    = np.array([
    [f"TP\n{tp}\n(Endo correctly found)",   f"FN\n{fn}\n(Endo missed)"],
    [f"FP\n{fp}\n(Healthy false alarm)",     f"TN\n{tn}\n(Healthy correctly cleared)"]
])

colors = np.array([
    ["#EAF3DE", "#FCEBEB"],
    ["#FAEEDA", "#E6F1FB"]
])
text_colors = np.array([
    ["#27500A", "#791F1F"],
    ["#633806", "#0C447C"]
])

for i in range(2):
    for j in range(2):
        ax.add_patch(plt.Rectangle(
            (j, 1-i), 1, 1,
            facecolor=colors[i][j], edgecolor="white", linewidth=3
        ))
        ax.text(j + 0.5, 1.5 - i, labels[i][j],
                ha="center", va="center",
                fontsize=12, color=text_colors[i][j],
                fontweight="bold" if i == j else "normal",
                linespacing=1.6)

ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5])
ax.set_xticklabels(["Predicted\nEndometriosis", "Predicted\nHealthy"], fontsize=11)
ax.set_yticks([0.5, 1.5])
ax.set_yticklabels(["Actual\nHealthy", "Actual\nEndometriosis"], fontsize=11)
ax.tick_params(length=0)

for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title(f"Confusion Matrix — Test Set (n = {tp+fn+fp+tn} patients)\n"
             f"Calibrated AdaBoost · Threshold = {optimal_thresh:.4f}",
             fontsize=13, fontweight="bold", pad=18)

summary = (f"Sensitivity: {sens*100:.1f}%  |  "
           f"Specificity: {spec*100:.1f}%  |  "
           f"Accuracy: {acc*100:.1f}%  |  "
           f"F1: {f1:.4f}")
fig.text(0.5, 0.01, summary, ha="center", fontsize=10, color="#555555")

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("confusion_matrix_final.png", dpi=150, bbox_inches="tight")
plt.close()
print(">> Saved: confusion_matrix_final.png")


# ════════════════════════════════════════════════════════════
# PLOT 3 — RADAR CHART
# ════════════════════════════════════════════════════════════
print("Generating radar chart...")

categories  = ["AUC", "CV-AUC", "Accuracy", "Sensitivity", "Specificity", "F1 Score"]
values_radar = [roc_auc_val, cv_auc, acc, sens, spec, f1]

N      = len(categories)
angles = [n / float(N) * 2 * math.pi for n in range(N)]
angles += angles[:1]

values_radar_plot = values_radar + values_radar[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
fig.patch.set_facecolor("white")

# Grid circles
for r in [0.6, 0.7, 0.8, 0.9, 1.0]:
    ax.plot(angles, [r] * (N + 1), color="#e5e7eb", linewidth=0.8, linestyle="--")
    ax.text(0, r, f"{r:.1f}", ha="center", va="center",
            fontsize=7, color="#9ca3af")

# Spoke lines
for angle in angles[:-1]:
    ax.plot([angle, angle], [0, 1], color="#e5e7eb", linewidth=0.8)

# Fill area
ax.fill(angles, values_radar_plot, color="#7c3aed", alpha=0.15)

# Border line
ax.plot(angles, values_radar_plot,
        color="#7c3aed", linewidth=2.5, linestyle="solid")

# Data points
# Data points
for angle, val in zip(angles[:-1], values_radar):
    ax.scatter(angle, val, color="#e11d48", s=80, zorder=5)
    ax.text(angle, val + 0.07, f"{val:.3f}",
            ha="center", va="center",
            fontsize=9, fontweight="bold", color="#7c3aed")

# Actually scatter correctly
for i, (angle, val) in enumerate(zip(angles[:-1], values_radar)):
    ax.scatter(angle, val, color="#e11d48", s=80, zorder=5)
    offset = 0.07
    ax.text(angle, val + offset, f"{val:.3f}",
            ha="center", va="center",
            fontsize=9, fontweight="bold", color="#7c3aed")

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11, fontweight="500", color="#1c0a2e")
ax.set_ylim(0, 1.1)
ax.set_yticks([])
ax.spines["polar"].set_visible(False)

ax.set_title("Model Performance Radar — FemCare\n"
             "Calibrated AdaBoost · Test Set",
             fontsize=13, fontweight="bold", pad=25, color="#1c0a2e")

purple_patch = mpatches.Patch(color="#7c3aed", alpha=0.4, label="AdaBoost (Calibrated)")
ax.legend(handles=[purple_patch], loc="upper right",
          bbox_to_anchor=(1.3, 1.15), fontsize=10)

plt.tight_layout()
plt.savefig("radar_final.png", dpi=150, bbox_inches="tight")
plt.close()
print(">> Saved: radar_final.png")

print("\n" + "=" * 55)
print("  ALL 3 PLOTS SAVED SUCCESSFULLY")
print("  roc_curve_final.png")
print("  confusion_matrix_final.png")
print("  radar_final.png")
print("=" * 55)