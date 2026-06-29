# ============================================================
# FEMCARE — PROPER FIX: PROBABILITY CALIBRATION
# Uses Platt Scaling (CalibratedClassifierCV) to fix
# AdaBoost's compressed probability outputs
# Saves calibrated model + threshold + validates tier coverage
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score,
                             f1_score, confusion_matrix, roc_curve)

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

print("=" * 60)
print("FEMCARE — PROBABILITY CALIBRATION FIX")
print("=" * 60)
print(f"Features: {len(FINAL_FEATURES)}")
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print()


# ── 1. ORIGINAL UNCALIBRATED MODEL (baseline) ────────────────
print("Training original AdaBoost (uncalibrated)...")
base_ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
base_ada.fit(X_train, y_train)
y_prob_uncal = base_ada.predict_proba(X_test)[:, 1]

# Check all-zeros patient on uncalibrated
all_zeros = pd.DataFrame([{f: 0 for f in FINAL_FEATURES}])
prob_zeros_uncal = base_ada.predict_proba(all_zeros)[0][1]

all_ones = pd.DataFrame([{f: 1 for f in FINAL_FEATURES}])
prob_ones_uncal = base_ada.predict_proba(all_ones)[0][1]

print(f"Uncalibrated — All symptoms absent : {prob_zeros_uncal:.4f} ({prob_zeros_uncal*100:.1f}%)")
print(f"Uncalibrated — All symptoms present: {prob_ones_uncal:.4f} ({prob_ones_uncal*100:.1f}%)")
print(f"Uncalibrated — Probability range   : {y_prob_uncal.min():.4f} to {y_prob_uncal.max():.4f}")
print()


# ── 2. CALIBRATED MODEL (Platt Scaling) ──────────────────────
print("Training calibrated AdaBoost (Platt Scaling)...")

# CalibratedClassifierCV with cv=5 uses cross-validation internally
# so it doesn't overfit the calibration to training data
calibrated_model = CalibratedClassifierCV(
    estimator=AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
    method='sigmoid',   # Platt scaling
    cv=5
)
calibrated_model.fit(X_train, y_train)
y_prob_cal = calibrated_model.predict_proba(X_test)[:, 1]

# Check all-zeros and all-ones on calibrated model
prob_zeros_cal = calibrated_model.predict_proba(all_zeros)[0][1]
prob_ones_cal  = calibrated_model.predict_proba(all_ones)[0][1]

print(f"Calibrated — All symptoms absent : {prob_zeros_cal:.4f} ({prob_zeros_cal*100:.1f}%)")
print(f"Calibrated — All symptoms present: {prob_ones_cal:.4f} ({prob_ones_cal*100:.1f}%)")
print(f"Calibrated — Probability range   : {y_prob_cal.min():.4f} to {y_prob_cal.max():.4f}")
print()


# ── 3. YOUDEN THRESHOLD ON CALIBRATED MODEL ──────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_prob_cal)
best_threshold = thresholds[np.argmax(tpr - fpr)]
y_pred_cal = (y_prob_cal >= best_threshold).astype(int)

print(f"Youden-optimal threshold (calibrated): {best_threshold:.4f}")
print()


# ── 4. METRICS COMPARISON ────────────────────────────────────
def get_metrics(y_test, y_prob, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'AUC'        : roc_auc_score(y_test, y_prob),
        'Accuracy'   : accuracy_score(y_test, y_pred),
        'Sensitivity': recall_score(y_test, y_pred),
        'Specificity': recall_score(y_test, y_pred, pos_label=0),
        'F1'         : f1_score(y_test, y_pred),
        'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn
    }

fpr_u, tpr_u, thresh_u = roc_curve(y_test, y_prob_uncal)
best_thresh_uncal = thresh_u[np.argmax(tpr_u - fpr_u)]
y_pred_uncal = (y_prob_uncal >= best_thresh_uncal).astype(int)

m_uncal = get_metrics(y_test, y_prob_uncal, y_pred_uncal)
m_cal   = get_metrics(y_test, y_prob_cal, y_pred_cal)

# CV-AUC for calibrated model
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc_cal = cross_val_score(calibrated_model, X, y, cv=cv, scoring='roc_auc')

print("=" * 60)
print("METRICS COMPARISON")
print("=" * 60)
print(f"{'Metric':<20} {'Uncalibrated':>14} {'Calibrated':>12}")
print("-" * 48)
metrics_to_show = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1']
for m in metrics_to_show:
    print(f"  {m:<18} {m_uncal[m]:>14.4f} {m_cal[m]:>12.4f}")
print(f"  {'CV-AUC (5-fold)':<18} {'—':>14} {cv_auc_cal.mean():>12.4f} ± {cv_auc_cal.std():.4f}")
print()
print(f"  {'TP (Endo found)':<18} {m_uncal['TP']:>14} {m_cal['TP']:>12}")
print(f"  {'FN (Endo missed)':<18} {m_uncal['FN']:>14} {m_cal['FN']:>12}")
print(f"  {'FP (False alarms)':<18} {m_uncal['FP']:>14} {m_cal['FP']:>12}")
print(f"  {'TN (Cleared)':<18} {m_uncal['TN']:>14} {m_cal['TN']:>12}")
print()


# ── 5. TIER VALIDATION ───────────────────────────────────────
print("=" * 60)
print("TIER COVERAGE — CALIBRATED MODEL")
print("=" * 60)

def assign_tier(prob):
    if prob >= 0.80:   return "Urgent"
    elif prob >= 0.60: return "High"
    elif prob >= 0.40: return "Moderate"
    else:              return "Low"

prob_df = pd.DataFrame({'probability': y_prob_cal, 'actual': y_test.values})
prob_df['tier'] = prob_df['probability'].apply(assign_tier)

tier_order = ['Low', 'Moderate', 'High', 'Urgent']
print(f"\n{'Tier':<12} {'Total':>7} {'Endo':>7} {'No-Endo':>9} {'Endo Rate':>11}")
print("-" * 50)
for tier in tier_order:
    subset   = prob_df[prob_df['tier'] == tier]
    total    = len(subset)
    endo     = int(subset['actual'].sum())
    no_endo  = total - endo
    rate     = endo / total if total > 0 else 0
    print(f"  {tier:<10} {total:>7} {endo:>7} {no_endo:>9} {rate:>10.1%}")

print()
print(f"All symptoms absent  → probability: {prob_zeros_cal*100:.1f}% → tier: {assign_tier(prob_zeros_cal)}")
print(f"All symptoms present → probability: {prob_ones_cal*100:.1f}% → tier: {assign_tier(prob_ones_cal)}")


# ── 6. CALIBRATION CURVE PLOT ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: calibration curves
prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, y_prob_uncal, n_bins=10)
prob_true_cal, prob_pred_cal     = calibration_curve(y_test, y_prob_cal, n_bins=10)

axes[0].plot([0,1], [0,1], 'k--', label='Perfectly calibrated')
axes[0].plot(prob_pred_uncal, prob_true_uncal, 'o-', color='#e74c3c',
             label='Uncalibrated AdaBoost', linewidth=2)
axes[0].plot(prob_pred_cal, prob_true_cal, 'o-', color='#2ecc71',
             label='Calibrated AdaBoost (Platt)', linewidth=2)
axes[0].set_xlabel('Mean Predicted Probability', fontsize=11)
axes[0].set_ylabel('Fraction of Positives', fontsize=11)
axes[0].set_title('Calibration Curve\n(closer to diagonal = better calibrated)',
                  fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# Right: probability distribution after calibration
endo_probs    = prob_df[prob_df['actual']==1]['probability']
no_endo_probs = prob_df[prob_df['actual']==0]['probability']

axes[1].hist(no_endo_probs, bins=20, alpha=0.6, color='#2ecc71',
             label='No Endometriosis', edgecolor='white')
axes[1].hist(endo_probs, bins=20, alpha=0.6, color='#e74c3c',
             label='Endometriosis', edgecolor='white')
for thresh, color, label in [(0.40, '#f39c12', '0.40 (Low/Moderate)'),
                              (0.60, '#e67e22', '0.60 (Moderate/High)'),
                              (0.80, '#c0392b', '0.80 (High/Urgent)')]:
    axes[1].axvline(thresh, color=color, linestyle='--', lw=2, label=label)
axes[1].set_xlabel('Calibrated Probability', fontsize=11)
axes[1].set_ylabel('Number of Patients', fontsize=11)
axes[1].set_title('Probability Distribution After Calibration',
                  fontsize=12, fontweight='bold')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('plot_calibration.png', dpi=150)
plt.close()
print()
print(">> Saved: plot_calibration.png")


# ── 7. SAVE CALIBRATED MODEL + THRESHOLD ─────────────────────
joblib.dump(calibrated_model, 'femcare_model_final.pkl')
joblib.dump(best_threshold, 'femcare_threshold_final.pkl')
print(">> Saved: femcare_model_final.pkl (calibrated)")
print(">> Saved: femcare_threshold_final.pkl")
print()
print("=" * 60)
print("DONE — femcare_model_final.pkl is now the calibrated model")
print("Backend loads this automatically — no other changes needed")
print("=" * 60)

import shap
explainer = shap.Explainer(calibrated_model.predict_proba, X_train, seed=42)
joblib.dump(explainer, 'femcare_explainer_final.pkl')
print(">> Saved: femcare_explainer_final.pkl (updated)")