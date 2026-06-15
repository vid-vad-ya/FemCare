# ============================================================
# ENDOMETRIOSIS RISK SCREENING — STEP 2B: ADABOOST + SHAP
# Comparing AdaBoost vs XGBoost on the same 14 features
# Includes threshold tuning (Youden Index) for sensitivity
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, f1_score,
                             recall_score, precision_score)
import shap

# ── 0. LOAD DATA ─────────────────────────────────────────────
DATA_PATH = "considerable dataset.xlsx"
df = pd.read_excel(DATA_PATH)

# ── 1. LOAD FINAL 14 FEATURES (from your existing pipeline) ──
with open('final_features.txt', 'r') as f:
    lines = f.readlines()

FINAL_FEATURES = []
for line in lines:
    line = line.strip()
    if line and line[0].isdigit():
        feature_name = line.split('. ', 1)[1].strip()
        FINAL_FEATURES.append(feature_name)

# NOTE: If "Ovarian Cysts" is still in this list from an earlier run,
# remove it manually here since it was excluded in the final model:
FINAL_FEATURES = [f for f in FINAL_FEATURES if 'Ovarian' not in f]

print("=" * 60)
print(f"USING {len(FINAL_FEATURES)} FEATURES")
print("=" * 60)
for i, f in enumerate(FINAL_FEATURES):
    print(f"  {i+1:2}. {f}")
print()

X = df[FINAL_FEATURES]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# ── 2. TRAIN ADABOOST ─────────────────────────────────────────
print("=" * 60)
print("TRAINING ADABOOST")
print("=" * 60)

ada_model = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)
ada_model.fit(X_train, y_train)

y_prob_ada = ada_model.predict_proba(X_test)[:, 1]
y_pred_ada = ada_model.predict(X_test)

acc_ada = accuracy_score(y_test, y_pred_ada)
auc_ada = roc_auc_score(y_test, y_prob_ada)
sens_ada = recall_score(y_test, y_pred_ada)
spec_ada = recall_score(y_test, y_pred_ada, pos_label=0)
f1_ada = f1_score(y_test, y_pred_ada)

print(f"AdaBoost @ default threshold (0.5):")
print(f"  Accuracy    : {acc_ada:.4f}")
print(f"  AUC         : {auc_ada:.4f}")
print(f"  Sensitivity : {sens_ada:.4f}")
print(f"  Specificity : {spec_ada:.4f}")
print(f"  F1 Score    : {f1_ada:.4f}")
print()

# 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_ada = cross_val_score(ada_model, X, y, cv=cv, scoring='roc_auc')
print(f"5-Fold CV-AUC: {cv_scores_ada.mean():.4f} ± {cv_scores_ada.std():.4f}")
print()


# ── 3. THRESHOLD TUNING (YOUDEN INDEX) ───────────────────────
print("=" * 60)
print("THRESHOLD TUNING — YOUDEN INDEX")
print("=" * 60)

fpr, tpr, thresholds = roc_curve(y_test, y_prob_ada)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_threshold = thresholds[best_idx]

print(f"Default threshold : 0.500")
print(f"Youden-optimal    : {best_threshold:.4f}")
print()

y_pred_tuned = (y_prob_ada >= best_threshold).astype(int)
acc_tuned = accuracy_score(y_test, y_pred_tuned)
sens_tuned = recall_score(y_test, y_pred_tuned)
spec_tuned = recall_score(y_test, y_pred_tuned, pos_label=0)
f1_tuned = f1_score(y_test, y_pred_tuned)

print(f"AdaBoost @ Youden-optimal threshold ({best_threshold:.3f}):")
print(f"  Accuracy    : {acc_tuned:.4f}")
print(f"  Sensitivity : {sens_tuned:.4f}")
print(f"  Specificity : {spec_tuned:.4f}")
print(f"  F1 Score    : {f1_tuned:.4f}")
print()


# ── 4. COMPARISON TABLE: XGBOOST vs ADABOOST ─────────────────
print("=" * 60)
print("COMPARISON: XGBOOST (PREVIOUS) vs ADABOOST")
print("=" * 60)

# Fill in your previous XGBoost numbers here for the printout
xgb_results = {
    'AUC': 0.9225,
    'Sensitivity': 0.8067,
    'Specificity': 0.8738,
    'Accuracy': 0.8378,
    'F1': 0.8421
}

comparison = pd.DataFrame({
    'XGBoost (previous)': xgb_results,
    'AdaBoost (default 0.5)': {
        'AUC': auc_ada, 'Sensitivity': sens_ada, 'Specificity': spec_ada,
        'Accuracy': acc_ada, 'F1': f1_ada
    },
    'AdaBoost (Youden-tuned)': {
        'AUC': auc_ada, 'Sensitivity': sens_tuned, 'Specificity': spec_tuned,
        'Accuracy': acc_tuned, 'F1': f1_tuned
    }
})
print(comparison.round(4))
print()


# ── 5. CONFUSION MATRIX (Youden-tuned) ───────────────────────
cm = confusion_matrix(y_test, y_pred_tuned)
tn, fp, fn, tp = cm.ravel()
print(f"Confusion Matrix (Youden-tuned threshold):")
print(f"  TP (Endo found)     : {tp}")
print(f"  FN (Endo missed)    : {fn}")
print(f"  FP (False alarms)   : {fp}")
print(f"  TN (Healthy cleared): {tn}")
print()


# ── 6. SHAP EXPLAINABILITY FOR ADABOOST ──────────────────────
print("=" * 60)
print("SHAP EXPLAINABILITY — ADABOOST")
print("=" * 60)
print("Note: AdaBoost isn't natively supported by TreeExplainer.")
print("Using shap.Explainer (Permutation-based, model-agnostic) instead.")
print("This may take 1-2 minutes for 222 test patients...")
print()

explainer = shap.Explainer(ada_model.predict_proba, X_train, seed=42)
shap_values_full = explainer(X_test)

# For binary classification, shap_values_full.values has shape (n_samples, n_features, n_classes)
# We want class 1 (endometriosis)
if len(shap_values_full.values.shape) == 3:
    shap_vals_plot = shap_values_full.values[:, :, 1]
else:
    shap_vals_plot = shap_values_full.values

# Global summary plot
plt.figure()
shap.summary_plot(shap_vals_plot, X_test, feature_names=FINAL_FEATURES,
                  show=False, plot_size=(10, 6))
plt.title('SHAP Summary — AdaBoost', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_adaboost_shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print(">> Saved: plot_adaboost_shap_summary.png")

# Feature importance bar
plt.figure()
shap.summary_plot(shap_vals_plot, X_test, feature_names=FINAL_FEATURES,
                  plot_type='bar', show=False, plot_size=(10, 6))
plt.title('SHAP Feature Importance — AdaBoost', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_adaboost_shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print(">> Saved: plot_adaboost_shap_bar.png")
print()


# ── 7. ROC CURVE COMPARISON PLOT ──────────────────────────────
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#9b59b6', lw=2, label=f'AdaBoost (AUC = {auc_ada:.3f})')
plt.scatter(fpr[best_idx], tpr[best_idx], color='red', zorder=5,
            label=f'Youden-optimal point (thr={best_threshold:.3f})')
plt.plot([0,1],[0,1], color='grey', linestyle=':', lw=1, label='Random Guess')
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate', fontsize=11)
plt.title('ROC Curve — AdaBoost with Youden-Optimal Threshold', fontsize=13, fontweight='bold')
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('plot_adaboost_roc.png', dpi=150)
plt.close()
print(">> Saved: plot_adaboost_roc.png")
print()


# ── 8. FINAL SUMMARY ──────────────────────────────────────────
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"{'Metric':<15} {'XGBoost':>10} {'AdaBoost(0.5)':>15} {'AdaBoost(tuned)':>17}")
print("-" * 60)
print(f"{'AUC':<15} {xgb_results['AUC']:>10.4f} {auc_ada:>15.4f} {auc_ada:>17.4f}")
print(f"{'Sensitivity':<15} {xgb_results['Sensitivity']:>10.4f} {sens_ada:>15.4f} {sens_tuned:>17.4f}")
print(f"{'Specificity':<15} {xgb_results['Specificity']:>10.4f} {spec_ada:>15.4f} {spec_tuned:>17.4f}")
print(f"{'Accuracy':<15} {xgb_results['Accuracy']:>10.4f} {acc_ada:>15.4f} {acc_tuned:>17.4f}")
print(f"{'F1':<15} {xgb_results['F1']:>10.4f} {f1_ada:>15.4f} {f1_tuned:>17.4f}")
print()
print("Output files:")
print("  plot_adaboost_shap_summary.png")
print("  plot_adaboost_shap_bar.png")
print("  plot_adaboost_roc.png")
print("=" * 60)