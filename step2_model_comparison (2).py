# ============================================================
# FEMCARE — STEP 2: MODEL COMPARISON + HYBRID + FEATURE ABLATION
# ============================================================
# Run: pip install xgboost  (first time only)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier,
                               VotingClassifier, GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_score)
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              roc_curve, confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBClassifier

# ── 0. LOAD DATA ─────────────────────────────────────────────
DATA_PATH = "dataset__1_.xlsx"   # <-- change path if needed
df = pd.read_excel(DATA_PATH)

EXCLUDE_COLS = ['Unnamed: 0', 'row', 'label']
feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
X_all = df[feature_cols]
y = df['label']

print("=" * 65)
print("FEMCARE — STEP 2: MODEL COMPARISON & HYBRID ENSEMBLE")
print("=" * 65)
print(f"Dataset: {len(df)} patients | {len(feature_cols)} symptoms")
print(f"Positive (Endo=1): {y.sum()} | Negative (0): {(y==0).sum()}\n")

# ── 1. FEATURE SELECTION (same pipeline as Step 1) ───────────
# 1a. Correlation filter
corr_matrix = X_all.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.85)]
X_filtered = X_all.drop(columns=to_drop)

# 1b. AdaBoost ranking
X_tr, X_te, y_tr, y_te = train_test_split(X_filtered, y, test_size=0.25,
                                            random_state=42, stratify=y)
ada_sel = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_sel.fit(X_tr, y_tr)
importances = pd.Series(ada_sel.feature_importances_, index=X_filtered.columns)
top_ada_features = importances[importances > 0].index.tolist()
X_ada = X_filtered[top_ada_features]

# 1c. RFE (15 features)
N_FEATURES = 15
rfe = RFE(estimator=LogisticRegression(max_iter=1000, random_state=42),
          n_features_to_select=N_FEATURES)
rfe.fit(X_ada, y)
final_features = [f for f, s in zip(top_ada_features, rfe.support_) if s]

print(f"Features after Selection Pipeline: {len(final_features)}")
print("Final 15 features:")
for i, f in enumerate(final_features):
    score = importances.get(f, 0)
    print(f"  {i+1:2}. {f:<55} (AdaBoost score: {score:.4f})")

top_feature = max(final_features, key=lambda f: importances.get(f, 0))
print(f"\n★ Top-weighted feature (will be ablated): '{top_feature}'\n")

X_final = X_all[final_features]

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.25, random_state=42, stratify=y)

# ── 2. MODEL DEFINITIONS ─────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "AdaBoost":            AdaBoostClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)":           SVC(kernel='rbf', probability=True, random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=100, use_label_encoder=False,
                                         eval_metric='logloss', random_state=42),
}

# ── 3. MODEL COMPARISON ──────────────────────────────────────
print("=" * 65)
print("SECTION A: MODEL COMPARISON (15 final features)")
print("=" * 65)

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n{'Model':<25} {'CV-AUC':>8} {'Test-AUC':>9} {'Acc':>7} {'F1':>7} {'Sens':>7} {'Spec':>7}")
print("-" * 70)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc_test = roc_auc_score(y_test, y_prob)
    acc      = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens     = tp / (tp + fn)
    spec     = tn / (tn + fp)

    cv_auc = cross_val_score(model, X_final, y, cv=cv, scoring='roc_auc').mean()

    results[name] = {
        'cv_auc': cv_auc, 'test_auc': auc_test, 'acc': acc,
        'f1': f1, 'sens': sens, 'spec': spec,
        'y_prob': y_prob, 'y_pred': y_pred,
    }
    print(f"  {name:<23} {cv_auc:>8.4f} {auc_test:>9.4f} {acc:>7.4f} "
          f"{f1:>7.4f} {sens:>7.4f} {spec:>7.4f}")

# ── 4. HYBRID VOTING ENSEMBLE ────────────────────────────────
print("\n" + "=" * 65)
print("SECTION B: HYBRID VOTING ENSEMBLE")
print("=" * 65)

sorted_models = sorted(results.items(), key=lambda x: x[1]['cv_auc'], reverse=True)
top3_names = [m[0] for m in sorted_models[:3]]
print(f"Top 3 models selected for ensemble: {top3_names}")

ensemble = VotingClassifier(
    estimators=[(name, models[name]) for name in top3_names],
    voting='soft'
)
ensemble.fit(X_train, y_train)
ens_pred = ensemble.predict(X_test)
ens_prob = ensemble.predict_proba(X_test)[:, 1]

ens_auc  = roc_auc_score(y_test, ens_prob)
ens_acc  = accuracy_score(y_test, ens_pred)
ens_f1   = f1_score(y_test, ens_pred)
cm_ens   = confusion_matrix(y_test, ens_pred)
tn, fp, fn, tp = cm_ens.ravel()
ens_sens = tp / (tp + fn)
ens_spec = tn / (tn + fp)
ens_cv   = cross_val_score(ensemble, X_final, y, cv=cv, scoring='roc_auc').mean()

results["Hybrid Ensemble"] = {
    'cv_auc': ens_cv, 'test_auc': ens_auc, 'acc': ens_acc,
    'f1': ens_f1, 'sens': ens_sens, 'spec': ens_spec,
    'y_prob': ens_prob, 'y_pred': ens_pred,
}

print(f"\n{'Model':<25} {'CV-AUC':>8} {'Test-AUC':>9} {'Acc':>7} {'F1':>7} {'Sens':>7} {'Spec':>7}")
print("-" * 70)
print(f"  {'Hybrid Ensemble':<23} {ens_cv:>8.4f} {ens_auc:>9.4f} {ens_acc:>7.4f} "
      f"{ens_f1:>7.4f} {ens_sens:>7.4f} {ens_spec:>7.4f}")

best_single = sorted_models[0]
print(f"\n  Best single model : {best_single[0]} (CV-AUC={best_single[1]['cv_auc']:.4f})")
delta = ens_cv - best_single[1]['cv_auc']
print(f"  Hybrid improvement: {'+' if delta>=0 else ''}{delta:.4f} AUC")

# ── 5. ABLATION STUDY ────────────────────────────────────────
print("\n" + "=" * 65)
print("SECTION C: ABLATION STUDY — FEATURE GROUP EXPERIMENTS")
print("=" * 65)

symptom_groups = {
    "Pain Group":               [f for f in final_features if any(w in f.lower() for w in
                                  ['pain', 'cramp', 'dysmenorrhea', 'dyspareunia', 'aching'])],
    "Bleeding Group":           [f for f in final_features if any(w in f.lower() for w in
                                  ['bleed', 'menstrual', 'menstruation', 'period', 'heavy'])],
    "GI Group":                 [f for f in final_features if any(w in f.lower() for w in
                                  ['bowel', 'ibs', 'diarrhea', 'constipation', 'digestive', 'nausea', 'vomit'])],
    "Systemic Group":           [f for f in final_features if any(w in f.lower() for w in
                                  ['fatigue', 'exhaustion', 'fever', 'anaemia', 'dizziness', 'anxiety', 'depression'])],
    "Without Top Feature":      [f for f in final_features if f != top_feature],
    "All 15 Features (Baseline)": final_features,
}

print("\nSymptom group membership:")
for gname, gfeats in symptom_groups.items():
    print(f"\n  [{gname}] ({len(gfeats)} features):")
    for f in gfeats:
        print(f"    - {f}")

print("\n\nAblation results using XGBoost:")
print(f"\n{'Feature Group':<35} {'# Feats':>7} {'CV-AUC':>8} {'Test-AUC':>9} {'F1':>7}")
print("-" * 70)

ablation_results = {}

for gname, gfeats in symptom_groups.items():
    if len(gfeats) == 0:
        print(f"  {gname:<33} — skipped (empty group)")
        continue

    Xg = X_all[gfeats]
    Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(
        Xg, y, test_size=0.25, random_state=42, stratify=y)

    xgb = XGBClassifier(n_estimators=100, use_label_encoder=False,
                         eval_metric='logloss', random_state=42)
    xgb.fit(Xg_tr, yg_tr)
    yg_prob = xgb.predict_proba(Xg_te)[:, 1]
    yg_pred = xgb.predict(Xg_te)

    auc_t = roc_auc_score(yg_te, yg_prob)
    f1_t  = f1_score(yg_te, yg_pred)
    cv_t  = cross_val_score(xgb, Xg, y, cv=cv, scoring='roc_auc').mean()

    ablation_results[gname] = {'cv_auc': cv_t, 'test_auc': auc_t, 'f1': f1_t, 'n': len(gfeats)}
    tag = " ← BASELINE" if gname == "All 15 Features (Baseline)" else ""
    print(f"  {gname:<33} {len(gfeats):>7} {cv_t:>8.4f} {auc_t:>9.4f} {f1_t:>7.4f}{tag}")

# ── 6. PLOTS ─────────────────────────────────────────────────

# Plot A: Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(13, 6))
model_names = list(results.keys())
cv_aucs   = [results[m]['cv_auc']   for m in model_names]
test_aucs  = [results[m]['test_auc'] for m in model_names]

x = np.arange(len(model_names))
w = 0.35
bars1 = ax.bar(x - w/2, cv_aucs,   w, label='CV AUC (5-fold)', color='#2980b9', alpha=0.85)
bars2 = ax.bar(x + w/2, test_aucs, w, label='Test AUC',        color='#e74c3c', alpha=0.85)

idx_hybrid = model_names.index("Hybrid Ensemble")
for b in [bars1[idx_hybrid], bars2[idx_hybrid]]:
    b.set_edgecolor('gold')
    b.set_linewidth(2.5)

ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('AUC Score', fontsize=11)
ax.set_ylim(0.5, 1.05)
ax.set_title('Model Comparison — CV AUC vs Test AUC\n(Gold border = Hybrid Ensemble)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.axhline(0.8, color='gray', linestyle='--', alpha=0.5)
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7.5)
plt.tight_layout()
plt.savefig('plot_A_model_comparison.png', dpi=150)
plt.close()
print("\n>> Saved: plot_A_model_comparison.png")

# Plot B: ROC Curves
fig, ax = plt.subplots(figsize=(9, 7))
colors_roc = ['#2980b9','#27ae60','#e67e22','#8e44ad','#e74c3c','#16a085','#f39c12']
for (name, res), col in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    lw = 2.5 if name == "Hybrid Ensemble" else 1.5
    ls = '-'  if name == "Hybrid Ensemble" else '--'
    ax.plot(fpr, tpr, color=col, lw=lw, linestyle=ls,
            label=f"{name} (AUC={res['test_auc']:.3f})")
ax.plot([0,1],[0,1],'k--', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curves — All Models (Hybrid = solid bold line)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('plot_B_roc_curves.png', dpi=150)
plt.close()
print(">> Saved: plot_B_roc_curves.png")

# Plot C: Radar Chart
metrics = ['CV AUC', 'Test AUC', 'Accuracy', 'F1', 'Sensitivity', 'Specificity']
keys    = ['cv_auc', 'test_auc', 'acc', 'f1', 'sens', 'spec']
N = len(metrics)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
colors_radar = ['#2980b9','#27ae60','#e67e22','#8e44ad','#e74c3c','#16a085','#f39c12']
for (name, res), col in zip(results.items(), colors_radar):
    vals = [res[k] for k in keys] + [res[keys[0]]]
    lw = 3 if name == "Hybrid Ensemble" else 1.5
    ax.plot(angles, vals, color=col, lw=lw, label=name)
    if name == "Hybrid Ensemble":
        ax.fill(angles, vals, color=col, alpha=0.15)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylim(0.5, 1.0)
ax.set_title('Model Performance Radar\n(Hybrid Ensemble highlighted)', fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8)
plt.tight_layout()
plt.savefig('plot_C_radar.png', dpi=150)
plt.close()
print(">> Saved: plot_C_radar.png")

# Plot D: Ablation Study
abl_names = list(ablation_results.keys())
abl_cv    = [ablation_results[n]['cv_auc'] for n in abl_names]
base_cv   = ablation_results.get("All 15 Features (Baseline)", {}).get('cv_auc', 0)

fig, ax = plt.subplots(figsize=(12, 6))
x2 = np.arange(len(abl_names))
abl_colors = ['#27ae60' if n == "All 15 Features (Baseline)"
              else '#e74c3c' if n == "Without Top Feature"
              else '#3498db' for n in abl_names]
bars = ax.bar(x2, abl_cv, color=abl_colors, edgecolor='white', linewidth=0.8)
ax.axhline(base_cv, color='green', linestyle='--', lw=1.5, alpha=0.7,
           label=f'Baseline CV AUC ({base_cv:.3f})')
ax.set_xticks(x2)
ax.set_xticklabels(abl_names, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('CV AUC (XGBoost)', fontsize=11)
ax.set_ylim(0.5, 1.05)
ax.set_title('Ablation Study — Feature Group Experiments\n(Green=Baseline | Red=Without Top Feature | Blue=Symptom Groups)',
             fontsize=13, fontweight='bold')
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', fontsize=9)
legend_patches = [
    mpatches.Patch(color='#27ae60', label='Baseline (all 15)'),
    mpatches.Patch(color='#e74c3c', label='Without Top Feature'),
    mpatches.Patch(color='#3498db', label='Symptom Group'),
]
ax.legend(handles=legend_patches, fontsize=9)
plt.tight_layout()
plt.savefig('plot_D_ablation.png', dpi=150)
plt.close()
print(">> Saved: plot_D_ablation.png")

# Plot E: Confusion Matrices
single_models = {k:v for k,v in results.items()
                 if k != "Hybrid Ensemble"}

best_name = sorted(single_models.items(),
                   key=lambda x: x[1]['sens'],
                   reverse=True)[0][0]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax_i, name in zip(axes, [best_name, "Hybrid Ensemble"]):
    cm = confusion_matrix(y_test, results[name]['y_pred'])
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Endo', 'Endo'])
    disp.plot(ax=ax_i, colorbar=False, cmap='Blues')
    ax_i.set_title(f'{name}\n(AUC={results[name]["test_auc"]:.3f})', fontsize=11, fontweight='bold')
plt.suptitle('Confusion Matrices — Best Single Model vs Hybrid Ensemble', fontsize=12)
plt.tight_layout()
plt.savefig('plot_E_confusion.png', dpi=150)
plt.close()
print(">> Saved: plot_E_confusion.png")

# ── 7. FINAL SUMMARY ─────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)

sorted_all = sorted(results.items(), key=lambda x: x[1]['cv_auc'], reverse=True)
print(f"\n{'Rank':<5} {'Model':<25} {'CV-AUC':>8} {'Test-AUC':>9} {'F1':>7} {'Sens':>7}")
print("-" * 60)
for rank, (name, res) in enumerate(sorted_all, 1):
    tag = " ★ BEST" if rank == 1 else ""
    print(f"  {rank:<4} {name:<25} {res['cv_auc']:>8.4f} {res['test_auc']:>9.4f} "
          f"{res['f1']:>7.4f} {res['sens']:>7.4f}{tag}")

print("\nAblation winner (highest CV-AUC group):")
best_abl = max(ablation_results.items(), key=lambda x: x[1]['cv_auc'])
print(f"  '{best_abl[0]}' — CV AUC={best_abl[1]['cv_auc']:.4f}")

print("\n" + "=" * 65)
print("STEP 2 COMPLETE")
print("Plots saved: plot_A through plot_E")
print("Next: Step 3 — SAFE comparison, Step 4 — SHAP explainability")
print("=" * 65)
