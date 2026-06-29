"""
femcare_evaluate.py
--------------------
Evaluates the exact model and features used by the FemCare frontend.

Place this file in the same folder as:
  - femcare_model.pkl
  - femcare_features.txt
  - considerable dataset.xlsx

Run:
  pip install scikit-learn pandas openpyxl joblib
  python femcare_evaluate.py
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# ─────────────────────────────────────────────
# 1. Load model  (exactly what main.py loads)
# ─────────────────────────────────────────────
model = joblib.load("femcare_model.pkl")
print("=" * 52)
print("  Model :", type(model).__name__)
print("  Params:", model.get_params())
print("=" * 52)

# ─────────────────────────────────────────────
# 2. Load features  (femcare_features.txt is the
#    single source of truth — same as main.py)
# ─────────────────────────────────────────────
with open("femcare_features.txt", "r") as f:
    FEATURES = [line.strip() for line in f if line.strip()]

print(f"\nFeatures used ({len(FEATURES)}):")
for i, feat in enumerate(FEATURES, 1):
    print(f"  {i:2}. {feat}")

# ─────────────────────────────────────────────
# 3. Load dataset
# ─────────────────────────────────────────────
df = pd.read_excel("considerable dataset.xlsx")
X  = df[FEATURES]
y  = df["label"]

print(f"\nDataset  : {len(df)} patients  |  Endo: {y.sum()}  |  Healthy: {(y==0).sum()}")

# ─────────────────────────────────────────────
# 4. Recreate the EXACT same train/test split
#    (same seed + stratify as the training script)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Split    : Train={len(X_train)}  Test={len(X_test)}")
print(f"Test set : Endo={y_test.sum()}  Healthy={(y_test==0).sum()}")

# ─────────────────────────────────────────────
# 5. Evaluate the SAVED model (do NOT retrain)
# ─────────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auc  = roc_auc_score(y_test, y_prob)
acc  = accuracy_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
sens = recall_score(y_test, y_pred)                  # sensitivity = recall for class 1
spec = recall_score(y_test, y_pred, pos_label=0)     # specificity = recall for class 0

# ─────────────────────────────────────────────
# 6. 5-Fold Cross-Validated AUC
# ─────────────────────────────────────────────
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
cv_auc  = cv_aucs.mean()

# ─────────────────────────────────────────────
# 7. Confusion Matrix
# ─────────────────────────────────────────────
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# ─────────────────────────────────────────────
# 8. Print Results
# ─────────────────────────────────────────────
print("\n" + "=" * 52)
print("  RESULTS  (femcare_model.pkl — production)")
print("=" * 52)
print(f"  AUC  (test set)      : {auc:.4f}")
print(f"  CV-AUC (5-fold mean) : {cv_auc:.4f}  {[round(s,4) for s in cv_aucs]}")
print(f"  Accuracy             : {acc*100:.2f}%")
print(f"  Sensitivity (Recall) : {sens*100:.2f}%")
print(f"  Specificity          : {spec*100:.2f}%")
print(f"  F1 Score             : {f1:.4f}")
print("=" * 52)

print("\n  CONFUSION MATRIX")
print(f"  {'':30} Predicted Endo   Predicted Healthy")
print(f"  {'Actual Endo':30} TP = {tp:<8}       FN = {fn}")
print(f"  {'Actual Healthy':30} FP = {fp:<8}       TN = {tn}")
print(f"\n  Total test patients  : {tp+fn+fp+tn}")
print(f"  Endo correctly found : {tp}  /  {tp+fn}")
print(f"  Endo missed          : {fn}  /  {tp+fn}")
print(f"  Healthy cleared      : {tn}  /  {fp+tn}")
print(f"  Healthy false-alarm  : {fp}  /  {fp+tn}")

print("\n  CLASSIFICATION REPORT")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Endometriosis"]))