# ============================================================
# Regenerate femcare_explainer.pkl in THIS environment.
#
# Why: the existing femcare_explainer.pkl was pickled by a
# different Python version than the one you're running now.
# shap.Explainer (permutation-based, since AdaBoost isn't a
# tree explainer target) embeds raw code objects, and the
# CodeType constructor signature differs across Python
# versions -> "code() argument 13 must be str, not int".
#
# This script rebuilds the explainer fresh, right here, so it
# matches your installed Python + shap + sklearn versions.
# Run it from the same folder as main.py (D:\FemCare):
#
#   py -3.11 regenerate_explainer.py
#
# It overwrites femcare_explainer.pkl. Then just restart main.py.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
import shap
from sklearn.model_selection import train_test_split

# ── 1. Load the feature list (same source of truth as main.py) ──
with open("femcare_features.txt", "r") as f:
    FEATURES = [line.strip() for line in f if line.strip()]

print(f"Using {len(FEATURES)} features (from femcare_features.txt)")

# ── 2. Load the same dataset used to train the model ──
DATA_PATH = "considerable dataset.xlsx"
df = pd.read_excel(DATA_PATH)

X = df[FEATURES]
y = df["label"]

# Same split params as the original training script (step2b_adaboost_shap.py)
# so the background data (X_train) matches what the model was tuned against.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ── 3. Load your already-trained model (this one loads fine) ──
model = joblib.load("femcare_model.pkl")

# ── 4. Rebuild the SHAP explainer fresh, in this environment ──
print("Building SHAP explainer (this can take 1-2 minutes)...")
explainer = shap.Explainer(model.predict_proba, X_train, seed=42)

# Sanity check it actually works before overwriting the old file
test_values = explainer(X_test.iloc[:3])
print("Sanity check OK — SHAP values shape:", test_values.values.shape)

# ── 5. Save it, overwriting the broken file ──
joblib.dump(explainer, "femcare_explainer.pkl")
print("Saved femcare_explainer.pkl successfully.")
print("Restart main.py and the explainer should load without errors.")