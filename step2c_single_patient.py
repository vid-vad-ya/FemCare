# ============================================================
# ENDOMETRIOSIS RISK SCREENING — STEP 2C: SINGLE PATIENT OUTPUT
# Shows risk %, tier, actual vs predicted, and SHAP explanation
# for one individual patient (AdaBoost, Youden-tuned threshold)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import shap

# ── 0. LOAD DATA ─────────────────────────────────────────────
DATA_PATH = "considerable dataset.xlsx"
df = pd.read_excel(DATA_PATH)

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


# ── 1. TRAIN ADABOOST + FIND YOUDEN THRESHOLD ────────────────
ada_model = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
ada_model.fit(X_train, y_train)

y_prob_test = ada_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob_test)
youden_j = tpr - fpr
best_threshold = thresholds[np.argmax(youden_j)]


# ── 2. RISK TIER FUNCTION ─────────────────────────────────────
def assign_risk_tier(prob):
    if prob >= 0.80:
        return "URGENT", "Consult a gynaecologist immediately"
    elif prob >= 0.60:
        return "HIGH", "Strongly recommend consultation"
    elif prob >= 0.40:
        return "MODERATE", "Schedule a check-up"
    else:
        return "LOW", "Monitor symptoms over time"


# ── 3. PICK A PATIENT TO EXPLAIN ──────────────────────────────
# Change this index (0 to len(X_test)-1) to view a different patient
PATIENT_INDEX = 0

patient_X = X_test.iloc[[PATIENT_INDEX]]
actual_label = y_test.iloc[PATIENT_INDEX]
patient_prob = ada_model.predict_proba(patient_X)[0][1]
predicted_label = int(patient_prob >= best_threshold)

tier, recommendation = assign_risk_tier(patient_prob)


# ── 4. PRINT PATIENT REPORT ────────────────────────────────────
print("=" * 60)
print(f"PATIENT REPORT — Test Patient #{PATIENT_INDEX + 1}")
print("=" * 60)
print()
print("Symptom Inputs:")
for feat in FINAL_FEATURES:
    val = patient_X.iloc[0][feat]
    status = "Yes" if val == 1 else "No"
    print(f"  {feat:<40}: {status}")
print()

print("-" * 60)
print("MODEL OUTPUT")
print("-" * 60)
print(f"  Risk Probability     : {patient_prob:.2%}")
print(f"  Risk Tier            : {tier}")
print(f"  Recommendation       : {recommendation}")
print(f"  Decision Threshold   : {best_threshold:.3f} (Youden-optimal)")
print()

print("-" * 60)
print("ACTUAL vs PREDICTED")
print("-" * 60)
actual_text = "Endometriosis (1)" if actual_label == 1 else "No Endometriosis (0)"
predicted_text = "Endometriosis (1)" if predicted_label == 1 else "No Endometriosis (0)"
match = "✅ CORRECT" if actual_label == predicted_label else "❌ INCORRECT"

print(f"  Actual (Ground Truth) : {actual_text}")
print(f"  Predicted (Model)     : {predicted_text}")
print(f"  Match                 : {match}")
print()


# ── 5. SHAP EXPLANATION FOR THIS PATIENT ──────────────────────
print("-" * 60)
print("WHY THE MODEL GAVE THIS SCORE (SHAP)")
print("-" * 60)
print("Calculating SHAP values for this patient... (this is quick for 1 patient)")
print()

explainer = shap.Explainer(ada_model.predict_proba, X_train, seed=42)
shap_values_patient = explainer(patient_X)

# Class 1 = endometriosis
if len(shap_values_patient.values.shape) == 3:
    shap_vals = shap_values_patient.values[0, :, 1]
    base_value = shap_values_patient.base_values[0, 1]
else:
    shap_vals = shap_values_patient.values[0]
    base_value = shap_values_patient.base_values[0]

# Sort by absolute impact
shap_series = pd.Series(shap_vals, index=FINAL_FEATURES)
shap_sorted = shap_series.reindex(shap_series.abs().sort_values(ascending=False).index)

print("Top symptoms driving this patient's score:")
print()
for feat, val in shap_sorted.head(5).items():
    direction = "↑ INCREASES risk" if val > 0 else "↓ DECREASES risk"
    present = "Present" if patient_X.iloc[0][feat] == 1 else "Absent"
    print(f"  {feat:<40} ({present:<7}) {direction}  (SHAP = {val:+.4f})")
print()


# ── 6. SHAP WATERFALL PLOT FOR THIS PATIENT ───────────────────
plt.figure(figsize=(10, 6))
shap.plots.waterfall(
    shap.Explanation(
        values=shap_vals,
        base_values=base_value,
        data=patient_X.iloc[0].values,
        feature_names=FINAL_FEATURES
    ),
    show=False
)
plt.title(f'SHAP Waterfall — Patient #{PATIENT_INDEX+1}\n'
          f'Risk: {patient_prob:.1%} ({tier}) | Actual: {actual_text} | Predicted: {predicted_text}',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'patient_{PATIENT_INDEX+1}_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()
print(f">> Saved: patient_{PATIENT_INDEX+1}_waterfall.png")
print()

print("=" * 60)
print("DONE")
print("=" * 60)
