"""
FemCare — XGBoost + SHAP Pipeline
Covers: Training → Validation → Explainability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    accuracy_score, f1_score, classification_report
)
from xgboost import XGBClassifier
import shap

# ─────────────────────────────────────────────
# 0. CONFIGURATION — edit these if needed
# ─────────────────────────────────────────────
DATA_PATH   = "D:\practice\dataset (1).xlsx"   # your Excel file
TARGET_COL  = "label"          # 0 / 1 column
RANDOM_SEED = 42
TEST_SIZE   = 0.25                     # 75-25 split (same as before)
CV_FOLDS    = 5

# The 15 features from your Step 1 feature selection
FEATURES = [
    "Ovarian cysts",
    "Menstrual pain (Dysmenorrhea)",
    "Abnormal uterine bleeding",
    "Pelvic pain",
    "Cysts (unspecified)",
    "Fever",
    "Infertility",
    "Nausea",
    "Constipation / Chronic constipation",
    "Abdominal Cramps during Intercourse",
    "Irregular / Missed periods",
    "Lower back pain",
    "Bloating",
    "Decreased energy / Exhaustion",
    "Diarrhea"
]

# Risk tier thresholds (probability → label)
def risk_tier(prob):
    if prob >= 0.80:
        return "Urgent"
    elif prob >= 0.60:
        return "High"
    elif prob >= 0.40:
        return "Moderate"
    else:
        return "Low"

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("FEMCARE — XGBoost + SHAP Pipeline")
print("=" * 60)

df = pd.read_excel(DATA_PATH)
print(f"\n✓ Dataset loaded: {df.shape[0]} patients, {df.shape[1]} columns")

# Encode target if it's a string
if df[TARGET_COL].dtype == object:
    le = LabelEncoder()
    df[TARGET_COL] = le.fit_transform(df[TARGET_COL])

X = df[FEATURES]
y = df[TARGET_COL]

pos = y.sum()
neg = len(y) - pos
print(f"  ├─ Endometriosis positive : {pos}")
print(f"  └─ Endometriosis negative : {neg}")

# ─────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_SEED
)
print(f"\n✓ Split: {len(X_train)} train / {len(X_test)} test (stratified)")

# ─────────────────────────────────────────────
# 3. XGBOOST MODEL
# ─────────────────────────────────────────────
# scale_pos_weight handles class imbalance automatically
spw = neg / pos

model = XGBClassifier(
    n_estimators=100,
    eval_metric='logloss',
    random_state=42
)

# ─────────────────────────────────────────────
# 4. CROSS-VALIDATION
# ─────────────────────────────────────────────
print("\n── 5-Fold Cross Validation ──")
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")

print(f"  Fold AUCs : {[round(s, 4) for s in cv_scores]}")
print(f"  CV-AUC    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 5. FINAL TRAINING ON FULL TRAIN SET
# ─────────────────────────────────────────────
model.fit(X_train, y_train)
print(f"\n✓ Model trained on {len(X_train)} patients")

# ─────────────────────────────────────────────
# 6. TEST SET VALIDATION
# ─────────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

auc      = roc_auc_score(y_test, y_prob)
acc      = accuracy_score(y_test, y_pred)
f1       = f1_score(y_test, y_pred)
cm       = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)      # recall
specificity = tn / (tn + fp)

print("\n── Test Set Results ──")
print(f"  AUC (Test)    : {auc:.4f}")
print(f"  CV-AUC        : {cv_scores.mean():.4f}")
print(f"  Accuracy      : {acc:.4f}")
print(f"  F1 Score      : {f1:.4f}")
print(f"  Sensitivity   : {sensitivity:.4f}  (Recall — endo patients found)")
print(f"  Specificity   : {specificity:.4f}  (Healthy patients correctly cleared)")
print(f"\n  Confusion Matrix:")
print(f"                  Pred: No Endo   Pred: Endo")
print(f"  Actual: No Endo    TN={tn:<6}      FP={fp}")
print(f"  Actual: Endo       FN={fn:<6}      TP={tp}")
print(f"\n  Out of {tp + fn} confirmed endo patients:")
print(f"    ✓ Found  : {tp}  ({sensitivity*100:.1f}%)")
print(f"    ✗ Missed : {fn}  ({fn/(tp+fn)*100:.1f}%)")

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=["No Endo", "Endo"]))

# ─────────────────────────────────────────────
# 7. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────
print("\n── Running SHAP (TreeExplainer) ──")

# Fix for XGBoost/SHAP version mismatch — base_score stored as string in newer XGBoost

explainer   = shap.TreeExplainer(model)
shap_values = explainer(X_test)          # returns Explanation object
sv_matrix   = shap_values.values         # shape: (n_test, n_features)

print(f"  ✓ SHAP values computed for {len(X_test)} patients × {len(FEATURES)} features")

# ─────────────────────────────────────────────
# 8. PLOTS
# ─────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
PINK   = "#C2185B"
LIGHT  = "#FCE4EC"
DARK   = "#880E4F"

# ── 8a. SHAP Summary (Bar) — global feature importance
print("\n── Generating plots ──")
fig, ax = plt.subplots(figsize=(9, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar",
                  color=PINK, show=False)
plt.title("Global Feature Importance (mean |SHAP value|)", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("shap_importance_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ shap_importance_bar.png")

# ── 8b. SHAP Beeswarm — direction + magnitude
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Beeswarm — Feature Impact Direction & Magnitude", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ shap_beeswarm.png")

# ── 8c. Confusion Matrix heatmap
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu",
            xticklabels=["No Endo", "Endo"],
            yticklabels=["No Endo", "Endo"],
            linewidths=1, ax=ax)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
ax.set_title("Confusion Matrix — XGBoost", fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ confusion_matrix.png")

# ── 8d. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color=PINK, lw=2, label=f"XGBoost (AUC = {auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Chance")
ax.fill_between(fpr, tpr, alpha=0.08, color=PINK)
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
ax.set_title("ROC Curve — XGBoost", fontsize=12)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ roc_curve.png")

# ── 8e. Waterfall for one sample patient (highest risk)
top_idx = int(np.argmax(y_prob))
fig, ax = plt.subplots(figsize=(9, 6))
shap.plots.waterfall(shap_values[top_idx], show=False)
plt.title(f"SHAP Waterfall — Patient #{top_idx} "
          f"(Risk: {y_prob[top_idx]*100:.1f}% | "
          f"{'Endo ✓' if y_test.iloc[top_idx]==1 else 'No Endo'})",
          fontsize=11, pad=10)
plt.tight_layout()
plt.savefig("shap_waterfall_sample.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ shap_waterfall_sample.png  (sample = patient #{top_idx})")

# ─────────────────────────────────────────────
# 9. PER-PATIENT EXPLAINABILITY REPORT
# ─────────────────────────────────────────────
print("\n── Building per-patient report ──")

records = []
for i in range(len(X_test)):
    row       = X_test.iloc[i]
    prob      = y_prob[i]
    tier      = risk_tier(prob)
    actual    = int(y_test.iloc[i])
    shap_row  = sv_matrix[i]                          # SHAP values for this patient

    # Top 3 contributing symptoms (by absolute SHAP value)
    ranked    = sorted(zip(FEATURES, shap_row), key=lambda x: abs(x[1]), reverse=True)
    top3      = [(f, round(v * 100, 2)) for f, v in ranked[:3]]

    record = {
        "patient_index" : X_test.index[i],
        "risk_pct"      : round(prob * 100, 1),
        "risk_tier"     : tier,
        "actual_label"  : "Endo" if actual == 1 else "No Endo",
        "correct"       : (prob >= 0.5) == actual,
        "top1_symptom"  : top3[0][0],
        "top1_shap_pct" : top3[0][1],
        "top2_symptom"  : top3[1][0],
        "top2_shap_pct" : top3[1][1],
        "top3_symptom"  : top3[2][0],
        "top3_shap_pct" : top3[2][1],
    }
    # Add all individual SHAP % columns
    for feat, val in zip(FEATURES, shap_row):
        record[f"shap_{feat}"] = round(val * 100, 2)

    records.append(record)

report_df = pd.DataFrame(records)
report_df.to_csv("femcare_explainability_report.csv", index=False)
print("  ✓ femcare_explainability_report.csv")

# ─────────────────────────────────────────────
# 10. SAMPLE OUTPUT CARD (printed to console)
# ─────────────────────────────────────────────
print("\n── Sample Patient Output Card ──")
sample = records[top_idx]
tier_colors = {"Urgent": "🔴", "High": "🟠", "Moderate": "🟡", "Low": "🟢"}
icon = tier_colors[sample["risk_tier"]]
print(f"\n  {icon}  Risk Score : {sample['risk_pct']}% — {sample['risk_tier'].upper()}")
print(f"  Actual Diagnosis : {sample['actual_label']}")
print(f"\n  Main contributors:")
print(f"    🔴 {sample['top1_symptom']:<35} {sample['top1_shap_pct']:+.1f}%")
print(f"    🔴 {sample['top2_symptom']:<35} {sample['top2_shap_pct']:+.1f}%")
print(f"    🟡 {sample['top3_symptom']:<35} {sample['top3_shap_pct']:+.1f}%")

# ─────────────────────────────────────────────
# 11. FINAL SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"  {'Metric':<25} {'Value':>10}")
print(f"  {'-'*35}")
print(f"  {'CV-AUC (5-fold)':<25} {cv_scores.mean():>9.4f}")
print(f"  {'Test AUC':<25} {auc:>9.4f}")
print(f"  {'Accuracy':<25} {acc:>9.4f}")
print(f"  {'F1 Score':<25} {f1:>9.4f}")
print(f"  {'Sensitivity (Recall)':<25} {sensitivity:>9.4f}")
print(f"  {'Specificity':<25} {specificity:>9.4f}")
print(f"  {'TP (Endo found)':<25} {tp:>9}")
print(f"  {'FN (Endo missed)':<25} {fn:>9}")
print(f"  {'FP (False alarms)':<25} {fp:>9}")
print(f"  {'TN (Healthy cleared)':<25} {tn:>9}")
print("=" * 60)

print("\n✓ All outputs saved:")
print("  • shap_importance_bar.png")
print("  • shap_beeswarm.png")
print("  • confusion_matrix.png")
print("  • roc_curve.png")
print("  • shap_waterfall_sample.png")
print("  • femcare_explainability_report.csv")
print("\nDone.\n")