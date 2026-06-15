# ============================================================
# ENDOMETRIOSIS RISK SCREENING — STEP 1: FEATURE SELECTION
# Pipeline: Correlation Filter → AdaBoost Ranking → RFE
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


DATA_PATH = "considerable dataset.xlsx"

df = pd.read_excel(DATA_PATH)

# Separate features and label
EXCLUDE_COLS = ['Unnamed: 0', 'row', 'label']
feature_cols = [col for col in df.columns if col not in EXCLUDE_COLS]

X = df[feature_cols]
y = df['label']

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total patients     : {len(df)}")
print(f"Total symptoms     : {len(feature_cols)}")
print(f"Endometriosis (1)  : {y.sum()} patients")
print(f"No Endometriosis(0): {(y == 0).sum()} patients")
print()


# ── 1. CORRELATION FILTER ────────────────────────────────────
# Remove features that are > 85% correlated with another feature
# (they carry duplicate information — no point keeping both)

print("=" * 60)
print("STEP 1: CORRELATION FILTER")
print("=" * 60)

corr_matrix = X.corr().abs()
upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.85)]

print(f"Highly correlated features removed: {to_drop if to_drop else 'None'}")
X_filtered = X.drop(columns=to_drop)
print(f"Features remaining: {X_filtered.shape[1]}")
print()

# Plot correlation heatmap (top 20 features for readability)
plt.figure(figsize=(14, 10))
top20 = X_filtered.iloc[:, :20]
sns.heatmap(top20.corr(), cmap='coolwarm', center=0,
            linewidths=0.3, annot=False)
plt.title('Correlation Heatmap (First 20 Features)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot1_correlation_heatmap.png', dpi=150)
plt.close()
print(">> Saved: plot1_correlation_heatmap.png")
print()


# ── 2. ADABOOST FEATURE RANKING ──────────────────────────────
# AdaBoost assigns an importance score to each feature
# Features with zero importance are noise — we drop them

print("=" * 60)
print("STEP 2: ADABOOST FEATURE RANKING")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y, test_size=0.25, random_state=42, stratify=y
)

ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)

importances = pd.Series(ada.feature_importances_, index=X_filtered.columns)
importances_sorted = importances.sort_values(ascending=False)

# Keep only features with non-zero importance
top_features_ada = importances[importances > 0].index.tolist()
print(f"Features with non-zero importance: {len(top_features_ada)}")
print()
print("Top 20 features ranked by AdaBoost importance:")
print("-" * 50)
for i, (feat, score) in enumerate(importances_sorted.head(20).items()):
    bar = "█" * int(score * 200)
    print(f"  {i+1:2}. {feat[:45]:<45} {score:.4f} {bar}")
print()

# Plot AdaBoost feature importances
plt.figure(figsize=(12, 8))
top15_ada = importances_sorted.head(15)
colors = ['#c0392b' if i < 5 else '#e67e22' if i < 10 else '#2980b9'
          for i in range(len(top15_ada))]
bars = plt.barh(range(len(top15_ada)), top15_ada.values[::-1], color=colors[::-1])
plt.yticks(range(len(top15_ada)),
           [f[:50] for f in top15_ada.index[::-1]], fontsize=9)
plt.xlabel('AdaBoost Feature Importance Score', fontsize=11)
plt.title('Top 15 Features — AdaBoost Ranking', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot2_adaboost_ranking.png', dpi=150)
plt.close()
print(">> Saved: plot2_adaboost_ranking.png")
print()

X_ada = X_filtered[top_features_ada]


# ── 3. RFE — RECURSIVE FEATURE ELIMINATION ───────────────────
# RFE progressively removes the weakest features one by one
# until we reach the desired number (15)

print("=" * 60)
print("STEP 3: RECURSIVE FEATURE ELIMINATION (RFE)")
print("=" * 60)

N_FEATURES = 15  # target number of final features

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_ada, y, test_size=0.25, random_state=42, stratify=y
)

rfe = RFE(
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    n_features_to_select=N_FEATURES
)
rfe.fit(X_train2, y_train2)

final_features = [f for f, s in zip(top_features_ada, rfe.support_) if s]
rfe_ranking = pd.Series(rfe.ranking_, index=top_features_ada).sort_values()

print(f"Features after RFE: {len(final_features)}")
print()


# ── 4. FINAL RESULTS ─────────────────────────────────────────

print("=" * 60)
print("FINAL SELECTED FEATURES")
print("=" * 60)

# Cross-reference with SAFE score features
SAFE_FEATURES = [
    'heavy', 'menstrual bleeding', 'dysmenorrhea', 'painful cramp',
    'pelvic pain', 'dyspareunia', 'painful sex', 'family history',
    'painkiller', 'treatment', 'irregular'
]

print(f"{'#':<4} {'Feature':<50} {'In SAFE?'}")
print("-" * 70)
for i, feat in enumerate(final_features):
    in_safe = any(s in feat.lower() for s in SAFE_FEATURES)
    tag = "✅ SAFE-aligned" if in_safe else "🆕 New signal"
    print(f"  {i+1:<3} {feat:<50} {tag}")

print()
print(f"Total final features: {len(final_features)}")
print()

# Save final feature list to a text file for reference
with open('final_features.txt', 'w') as f:
    f.write("FINAL SELECTED FEATURES FOR ENDOMETRIOSIS RISK MODEL\n")
    f.write("=" * 55 + "\n\n")
    for i, feat in enumerate(final_features):
        f.write(f"{i+1}. {feat}\n")
print(">> Saved: final_features.txt")
print()


# ── 5. FEATURE SELECTION SUMMARY PLOT ────────────────────────

fig, ax = plt.subplots(figsize=(12, 7))
colors = ['#27ae60' if any(s in f.lower() for s in SAFE_FEATURES)
          else '#2980b9' for f in final_features]
ada_scores = [importances.get(f, 0) for f in final_features]
sorted_idx = np.argsort(ada_scores)
sorted_features = [final_features[i] for i in sorted_idx]
sorted_scores = [ada_scores[i] for i in sorted_idx]
sorted_colors = [colors[i] for i in sorted_idx]

bars = ax.barh(range(len(sorted_features)), sorted_scores,
               color=sorted_colors, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(sorted_features)))
ax.set_yticklabels([f[:52] for f in sorted_features], fontsize=9)
ax.set_xlabel('AdaBoost Importance Score', fontsize=11)
ax.set_title('Final 15 Selected Features\n(Green = SAFE-aligned | Blue = New signal)',
             fontsize=13, fontweight='bold')

from matplotlib.patches import Patch
legend = [Patch(color='#27ae60', label='SAFE-aligned feature'),
          Patch(color='#2980b9', label='New signal (not in SAFE)')]
ax.legend(handles=legend, loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig('plot3_final_features.png', dpi=150)
plt.close()
print(">> Saved: plot3_final_features.png")
print()


