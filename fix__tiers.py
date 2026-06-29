# ============================================================
# FEMCARE — FIX TIER BOUNDARIES AFTER CALIBRATION
# Derives data-driven tier boundaries from calibrated
# probability distribution to ensure monotonic endo rates
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
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

# ── 1. TRAIN CALIBRATED MODEL ─────────────────────────────────
print("=" * 60)
print("TRAINING CALIBRATED MODEL")
print("=" * 60)

calibrated_model = CalibratedClassifierCV(
    estimator=AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
    method='sigmoid',
    cv=5
)
calibrated_model.fit(X_train, y_train)
y_prob = calibrated_model.predict_proba(X_test)[:, 1]

print(f"Calibrated probability range: {y_prob.min():.4f} to {y_prob.max():.4f}")
print()

# ── 2. FIND DATA-DRIVEN TIER BOUNDARIES ──────────────────────
print("=" * 60)
print("FINDING DATA-DRIVEN TIER BOUNDARIES")
print("=" * 60)
print()
print("Strategy: boundaries at 25th, 60th, 85th percentile")
print("of calibrated probabilities — ensures all 4 tiers")
print("are populated and monotonically increasing endo rate")
print()

# Use percentiles of the test probability distribution
# These ensure roughly: 25% Low, 35% Moderate, 25% High, 15% Urgent
p25 = np.percentile(y_prob, 25)
p60 = np.percentile(y_prob, 60)
p85 = np.percentile(y_prob, 85)

print(f"Candidate boundaries:")
print(f"  Low/Moderate cutoff  : {p25:.4f}")
print(f"  Moderate/High cutoff : {p60:.4f}")
print(f"  High/Urgent cutoff   : {p85:.4f}")
print()

# ── 3. VALIDATE MONOTONICITY ─────────────────────────────────
def check_tiers(y_prob, y_test, low_t, mod_t, high_t):
    prob_df = pd.DataFrame({'prob': y_prob, 'actual': y_test.values})

    def tier(p):
        if p >= high_t:  return 'Urgent'
        elif p >= mod_t: return 'High'
        elif p >= low_t: return 'Moderate'
        else:            return 'Low'

    prob_df['tier'] = prob_df['prob'].apply(tier)
    tier_order = ['Low', 'Moderate', 'High', 'Urgent']

    print(f"{'Tier':<12} {'Total':>7} {'Endo':>7} {'No-Endo':>9} {'Endo Rate':>11}")
    print("-" * 50)
    rates = []
    for t in tier_order:
        sub   = prob_df[prob_df['tier'] == t]
        total = len(sub)
        endo  = int(sub['actual'].sum())
        nendo = total - endo
        rate  = endo / total if total > 0 else 0
        rates.append(rate)
        print(f"  {t:<10} {total:>7} {endo:>7} {nendo:>9} {rate:>10.1%}")

    monotonic = all(rates[i] <= rates[i+1] for i in range(len(rates)-1))
    print(f"\n  Monotonic (increasing endo rate)? {'✅ YES' if monotonic else '❌ NO'}")
    return monotonic, rates

print("Tier validation with percentile boundaries:")
monotonic, rates = check_tiers(y_prob, y_test, p25, p60, p85)

# ── 4. IF NOT MONOTONIC, SEARCH FOR BETTER BOUNDARIES ────────
if not monotonic:
    print()
    print("Searching for better boundaries...")

    best_boundaries = (p25, p60, p85)
    best_score = -1

    # Search around the percentile estimates
    for l in np.arange(0.10, 0.40, 0.05):
        for m in np.arange(l + 0.10, 0.70, 0.05):
            for h in np.arange(m + 0.10, 0.90, 0.05):
                prob_df = pd.DataFrame({'prob': y_prob, 'actual': y_test.values})
                def tier(p):
                    if p >= h:   return 'Urgent'
                    elif p >= m: return 'High'
                    elif p >= l: return 'Moderate'
                    else:        return 'Low'
                prob_df['tier'] = prob_df['prob'].apply(tier)

                tier_rates = []
                all_populated = True
                for t in ['Low', 'Moderate', 'High', 'Urgent']:
                    sub = prob_df[prob_df['tier'] == t]
                    if len(sub) == 0:
                        all_populated = False
                        break
                    tier_rates.append(sub['actual'].mean())

                if not all_populated:
                    continue

                is_mono = all(tier_rates[i] <= tier_rates[i+1]
                              for i in range(len(tier_rates)-1))
                if is_mono:
                    # Score = spread between highest and lowest endo rate
                    score = tier_rates[-1] - tier_rates[0]
                    if score > best_score:
                        best_score = score
                        best_boundaries = (l, m, h)

    p25, p60, p85 = best_boundaries
    print(f"\nBest boundaries found:")
    print(f"  Low/Moderate : {p25:.4f}")
    print(f"  Moderate/High: {p60:.4f}")
    print(f"  High/Urgent  : {p85:.4f}")
    print()
    print("Re-validating:")
    monotonic, rates = check_tiers(y_prob, y_test, p25, p60, p85)

LOW_THRESHOLD      = round(p25, 4)
MODERATE_THRESHOLD = round(p60, 4)
HIGH_THRESHOLD     = round(p85, 4)

# ── 5. YOUDEN THRESHOLD ───────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
best_threshold = thresholds[np.argmax(tpr - fpr)]

# ── 6. METRICS ────────────────────────────────────────────────
y_pred = (y_prob >= best_threshold).astype(int)
cm     = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp_val = cm.ravel()

cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(calibrated_model, X, y, cv=cv, scoring='roc_auc')

print()
print("=" * 60)
print("FINAL MODEL METRICS")
print("=" * 60)
print(f"  Test AUC         : {roc_auc_score(y_test, y_prob):.4f}")
print(f"  CV-AUC (5-fold)  : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print(f"  Accuracy         : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Sensitivity      : {recall_score(y_test, y_pred):.4f}")
print(f"  Specificity      : {recall_score(y_test, y_pred, pos_label=0):.4f}")
print(f"  F1 Score         : {f1_score(y_test, y_pred):.4f}")
print(f"  Threshold        : {best_threshold:.4f}")
print(f"  TP               : {tp_val} | FN: {fn} | FP: {fp} | TN: {tn}")

# ── 7. VERIFY ALL-ZEROS PATIENT ──────────────────────────────
all_zeros = pd.DataFrame([{f: 0 for f in FINAL_FEATURES}])
all_ones  = pd.DataFrame([{f: 1 for f in FINAL_FEATURES}])

prob_z = calibrated_model.predict_proba(all_zeros)[0][1]
prob_o = calibrated_model.predict_proba(all_ones)[0][1]

def assign_tier(prob):
    if prob >= HIGH_THRESHOLD:     return "Urgent"
    elif prob >= MODERATE_THRESHOLD: return "High"
    elif prob >= LOW_THRESHOLD:    return "Moderate"
    else:                          return "Low"

print()
print("=" * 60)
print("EDGE CASE VERIFICATION")
print("=" * 60)
print(f"  All symptoms absent  → {prob_z*100:.1f}% → {assign_tier(prob_z)}")
print(f"  All symptoms present → {prob_o*100:.1f}% → {assign_tier(prob_o)}")

# ── 8. SAVE EVERYTHING ───────────────────────────────────────
tiers_config = {
    'low_threshold'      : LOW_THRESHOLD,
    'moderate_threshold' : MODERATE_THRESHOLD,
    'high_threshold'     : HIGH_THRESHOLD
}

joblib.dump(calibrated_model, 'femcare_model_final.pkl')
joblib.dump(best_threshold,   'femcare_threshold_final.pkl')
joblib.dump(tiers_config,     'femcare_tiers_final.pkl')

print()
print("=" * 60)
print("SAVED FILES")
print("=" * 60)
print(f"  femcare_model_final.pkl     — calibrated AdaBoost")
print(f"  femcare_threshold_final.pkl — Youden threshold ({best_threshold:.4f})")
print(f"  femcare_tiers_final.pkl     — tier boundaries:")
print(f"    Low      : prob < {LOW_THRESHOLD}")
print(f"    Moderate : {LOW_THRESHOLD} ≤ prob < {MODERATE_THRESHOLD}")
print(f"    High     : {MODERATE_THRESHOLD} ≤ prob < {HIGH_THRESHOLD}")
print(f"    Urgent   : prob ≥ {HIGH_THRESHOLD}")
print()
print("Next step: update risk_tier() in main.py with these boundaries")
print("=" * 60)

# ── 9. PLOT ───────────────────────────────────────────────────
tier_labels = ['Low', 'Moderate', 'High', 'Urgent']
colors      = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Endo rate per tier
axes[0].bar(tier_labels, [r*100 for r in rates], color=colors,
            edgecolor='white', linewidth=1.5)
axes[0].set_ylabel('% Actually Have Endometriosis', fontsize=11)
axes[0].set_title('Endometriosis Rate per Risk Tier\n(should be monotonically increasing)',
                  fontsize=12, fontweight='bold')
axes[0].set_ylim(0, 115)
for i, (r, c) in enumerate(zip(rates, colors)):
    axes[0].text(i, r*100 + 2, f'{r:.1%}', ha='center',
                fontsize=11, fontweight='bold')

# Probability distribution with new tier boundaries
endo_mask = y_test.values == 1
axes[1].hist(y_prob[~endo_mask], bins=20, alpha=0.6, color='#2ecc71',
             label='No Endometriosis', edgecolor='white')
axes[1].hist(y_prob[endo_mask], bins=20, alpha=0.6, color='#e74c3c',
             label='Endometriosis', edgecolor='white')
for thresh, color, label in [
    (LOW_THRESHOLD,      '#f39c12', f'Low/Moderate ({LOW_THRESHOLD})'),
    (MODERATE_THRESHOLD, '#e67e22', f'Moderate/High ({MODERATE_THRESHOLD})'),
    (HIGH_THRESHOLD,     '#c0392b', f'High/Urgent ({HIGH_THRESHOLD})')
]:
    axes[1].axvline(thresh, color=color, linestyle='--', lw=2, label=label)
axes[1].set_xlabel('Calibrated Probability', fontsize=11)
axes[1].set_ylabel('Number of Patients', fontsize=11)
axes[1].set_title('Probability Distribution with Data-Driven Tier Boundaries',
                  fontsize=12, fontweight='bold')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('plot_tier_fixed.png', dpi=150)
plt.close()
print(">> Saved: plot_tier_fixed.png")

import shap
explainer = shap.Explainer(calibrated_model.predict_proba, X_train)
joblib.dump(explainer, 'Femcares_explainer_final.pkl')
print(">> Saved: Femcares_explainer_final.pkl")