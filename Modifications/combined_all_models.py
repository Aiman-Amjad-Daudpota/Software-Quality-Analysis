"""
================================================================
COMBINED PIPELINE — ALL 6 MODELS × 3 EXPERIMENTAL CONDITIONS
================================================================
Models (6):
  Original paper : Logistic Regression, Decision Tree, Random Forest
  Extended       : k-NN, XGBoost, Linear SVC

Experimental Conditions (3):
  Exp A — Baseline      : Original data (10,885), all 21 features
  Exp B — After Removal : Cleaned data  (8,912),  all 21 features
  Exp C — Removal + FS  : Cleaned data  (8,912),  top 7 features

Output:
  - Saved model files (.joblib) for all 18 trained models
  - Weighted evaluation metrics for all 18 experiments
  - Combined comparison table and plots
================================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import StandardScaler
from sklearn.pipeline           import Pipeline
from sklearn.linear_model       import LogisticRegression
from sklearn.tree               import DecisionTreeClassifier
from sklearn.ensemble           import RandomForestClassifier
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.svm                import LinearSVC
from sklearn.calibration        import CalibratedClassifierCV
from sklearn.metrics            import (classification_report,
                                        confusion_matrix,
                                        roc_auc_score, roc_curve)
from imblearn.over_sampling     import SMOTE
from xgboost                    import XGBClassifier

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120

os.makedirs('plots',  exist_ok=True)
os.makedirs('models', exist_ok=True)

# ─── Features ────────────────────────────────────────────────────────────────
ALL_FEATS = ['loc','v(g)','ev(g)','iv(g)','n','v','l','d','i','e','b','t',
             'lOCode','lOComment','lOBlank','locCodeAndComment',
             'uniq_Op','uniq_Opnd','total_Op','total_Opnd','branchCount']
TOP7      = ['loc','branchCount','v(g)','v','d','e','lOCode']

# ─── Colours ─────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    'Logistic Regression': '#1565C0',
    'Decision Tree':       '#2E7D32',
    'Random Forest':       '#558B2F',
    'k-NN':                '#F57F17',
    'XGBoost':             '#BF360C',
    'Linear SVC':          '#6A1B9A',
}
EXP_LABELS = {
    'A': 'A — Baseline',
    'B': 'B — After Removal',
    'C': 'C — Removal + FS',
}

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD & PREPARE THREE DATASET CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("="*72)
print("LOADING DATASETS")
print("="*72)

# ── Exp A: original (all 10,885 rows, all 21 features) ──────────────────────
raw = pd.read_csv('../Data Cleaning/jm1.csv')
for col in ALL_FEATS:
    raw[col] = pd.to_numeric(raw[col], errors='coerce')
raw['defects_encoded'] = raw['defects'].map(
    {True:1, False:0, 'True':1, 'False':0}).astype(float)
raw = raw.dropna(subset=['defects_encoded'])
for col in ALL_FEATS:
    raw[col] = raw[col].fillna(raw[col].median())
raw['defects_encoded'] = raw['defects_encoded'].astype(int)

Xa, ya = raw[ALL_FEATS], raw['defects_encoded']

# ── Exp B/C: cleaned (8,912 rows) ────────────────────────────────────────────
cl = pd.read_csv('../Data Cleaning/jm1_cleaned.csv')
Xb, yb = cl[ALL_FEATS], cl['defects_encoded']
Xc, yc = cl[TOP7],      cl['defects_encoded']

datasets = {
    'A': (Xa, ya, f"Baseline (n={len(raw):,}, 21 feats)"),
    'B': (Xb, yb, f"After Removal (n={len(cl):,}, 21 feats)"),
    'C': (Xc, yc, f"Removal + FS  (n={len(cl):,}, 7 feats)"),
}
for k, (X, y, desc) in datasets.items():
    print(f"  Exp {k}: {desc} | class balance: "
          f"{(y==0).sum():,} clean / {(y==1).sum():,} defective")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
def make_models(pos_ratio):
    return {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=1000, random_state=42,
                class_weight='balanced', solver='lbfgs'))
        ]),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20,
            class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1),
        'k-NN': KNeighborsClassifier(
            n_neighbors=7, metric='euclidean', n_jobs=-1),
        'XGBoost': XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            scale_pos_weight=pos_ratio, random_state=42,
            eval_metric='logloss', verbosity=0),
        'Linear SVC': CalibratedClassifierCV(
            LinearSVC(max_iter=2000, random_state=42), cv=5),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL 18 EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("RUNNING 18 EXPERIMENTS (6 models × 3 conditions)")
print("="*72)

results = []

for exp_key, (X, y, desc) in datasets.items():
    exp_label = EXP_LABELS[exp_key]

    # 80/20 stratified split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    # SMOTE on training only
    smote = SMOTE(random_state=42)
    X_tr_s, y_tr_s = smote.fit_resample(X_tr, y_tr)

    # Scale (fit on SMOTE'd training, apply to test)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_s)
    X_te_sc  = scaler.transform(X_te)

    # Save scaler for this experiment
    joblib.dump(scaler,
        f'models/scaler_exp{exp_key}.joblib')

    pos_ratio = (y_tr == 0).sum() / (y_tr == 1).sum()
    models    = make_models(pos_ratio)

    for mname, model in models.items():
        print(f"  Exp {exp_key} | {mname:22s} ... ", end='', flush=True)

        model.fit(X_tr_sc, y_tr_s)

        # Save model file
        safe_name = mname.lower().replace(' ', '_').replace('-', '')
        model_path = f'models/{safe_name}_exp{exp_key}.joblib'
        joblib.dump(model, model_path)

        y_pred = model.predict(X_te_sc)
        y_prob = model.predict_proba(X_te_sc)[:, 1]

        rep   = classification_report(y_te, y_pred,
                                      output_dict=True, zero_division=0)
        auc   = roc_auc_score(y_te, y_prob)
        cm    = confusion_matrix(y_te, y_pred)
        cm_n  = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        # Original-paper style: minority-class (defective) metrics
        def_prec = rep.get('1', {}).get('precision', 0)
        def_rec  = rep.get('1', {}).get('recall',    0)
        def_f1   = rep.get('1', {}).get('f1-score',  0)

        r = {
            'Experiment':   exp_label,
            'Exp_key':      exp_key,
            'Model':        mname,
            'Model_file':   model_path,
            # Accuracy
            'Accuracy':     rep['accuracy'],
            # Weighted (what professor asked for)
            'W-Precision':  rep['weighted avg']['precision'],
            'W-Recall':     rep['weighted avg']['recall'],
            'W-F1':         rep['weighted avg']['f1-score'],
            'ROC-AUC':      auc,
            # Defective-class only (for comparison with original paper)
            'Def-Precision':def_prec,
            'Def-Recall':   def_rec,
            'Def-F1':       def_f1,
            # Raw data for plots
            'cm':           cm,
            'cm_norm':      cm_n,
            'y_te':         y_te,
            'y_prob':       y_prob,
        }
        results.append(r)
        print(f"AUC={auc:.4f}  W-F1={rep['weighted avg']['f1-score']:.4f}"
              f"  → saved {model_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# ITEM 8 — ANSWER + COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("ITEM 8 — ORIGINAL PAPER vs WEIGHTED METRICS COMPARISON")
print("="*72)

print("""
  The original paper (Table VII) reported Precision and Recall for the
  DEFECTIVE (minority) class only — NOT weighted averages. Evidence:
    LR  : Precision=0.42, Recall=0.58
    DT  : Precision=0.38, Recall=0.52
    RF  : Precision=0.46, Recall=0.55
  With 77.5% non-defective modules, true weighted precision would be
  ~0.73–0.79.  The low values (0.38–0.46) are a clear signature of
  minority-class-only reporting.

  Weighted metrics weight each class by its proportion in the test set:
    Weighted Precision = Σ (class_precision × class_support) / total_support
  This gives a single number that accounts for class imbalance and
  represents overall model quality across BOTH classes.
""")

# Side-by-side for original 3 models, Exp B (matching paper's cleaned dataset)
print("  Original paper vs weighted — Exp B (After Removal, matching paper):\n")
paper_vals = {
    'Logistic Regression': (0.42, 0.58, 0.689),
    'Decision Tree':       (0.38, 0.52, 0.650),
    'Random Forest':       (0.46, 0.55, 0.694),
}
print(f"  {'Model':<22} {'Def-Prec':>10} {'Def-Rec':>9} "
      f"{'W-Prec':>9} {'W-Rec':>9} {'W-F1':>8} {'AUC':>8}")
print("  " + "-"*78)
for mname, (op, or_, oa) in paper_vals.items():
    r = next((x for x in results
              if x['Model']==mname and x['Exp_key']=='B'), None)
    if r:
        print(f"  {mname:<22} "
              f"{op:>10.2f} {or_:>9.2f} "
              f"{r['W-Precision']:>9.4f} {r['W-Recall']:>9.4f} "
              f"{r['W-F1']:>8.4f} {r['ROC-AUC']:>8.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# ITEM 9 — FULL WEIGHTED RESULTS TABLE (all 18)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("ITEM 9 — FULL WEIGHTED EVALUATION — ALL 18 EXPERIMENTS")
print("="*72)

df_all = pd.DataFrame([{
    'Experiment':  r['Experiment'],
    'Model':       r['Model'],
    'Accuracy':    f"{r['Accuracy']*100:.1f}%",
    'W-Precision': f"{r['W-Precision']:.4f}",
    'W-Recall':    f"{r['W-Recall']:.4f}",
    'W-F1':        f"{r['W-F1']:.4f}",
    'ROC-AUC':     f"{r['ROC-AUC']:.4f}",
    'Def-Precision': f"{r['Def-Precision']:.4f}",
    'Def-Recall':    f"{r['Def-Recall']:.4f}",
} for r in results])

print(df_all.to_string(index=False))
df_all.to_csv('combined_evaluation_results.csv', index=False)
print("\nSaved: combined_evaluation_results.csv")

best_auc = max(results, key=lambda r: r['ROC-AUC'])
best_f1  = max(results, key=lambda r: r['W-F1'])
print(f"\nBest ROC-AUC : {best_auc['Model']} | {best_auc['Experiment']} "
      f"(AUC={best_auc['ROC-AUC']:.4f})")
print(f"Best W-F1    : {best_f1['Model']}  | {best_f1['Experiment']} "
      f"(W-F1={best_f1['W-F1']:.4f})")

# ═══════════════════════════════════════════════════════════════════════════════
# ITEM 10 — NORMALISED CONFUSION MATRIX for best AUC model
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print(f"ITEM 10 — Normalised CM: {best_auc['Model']} | {best_auc['Experiment']}")
print("="*72)
print("Raw:\n", best_auc['cm'])
print("Normalised:\n", np.round(best_auc['cm_norm'], 4))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, data, title, fmt in zip(
    axes,
    [best_auc['cm'], best_auc['cm_norm']],
    ['Raw Counts', 'Normalised (by true class)'],
    ['d', '.2%']
):
    sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                xticklabels=['Non-Defective','Defective'],
                yticklabels=['Non-Defective','Defective'],
                linewidths=0.5, annot_kws={'size':11})
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label',      fontsize=10)
    ax.set_title(f"{title}\n{best_auc['Model']} | {best_auc['Experiment']}",
                 fontsize=10)
plt.tight_layout()
plt.savefig('plots/combined_item10_normalised_cm.png', bbox_inches='tight')
plt.close()
print("Saved: plots/combined_item10_normalised_cm.png")

# ═══════════════════════════════════════════════════════════════════════════════
# ITEM 11 — ROC CURVES — all 18 experiments (3 panels)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
model_order = ['Logistic Regression','Decision Tree','Random Forest',
               'k-NN','XGBoost','Linear SVC']
linestyles  = ['-','-','-','--','--','--']   # solid=original, dashed=new

for ax, exp_key in zip(axes, ['A','B','C']):
    exp_label = EXP_LABELS[exp_key]
    exp_res   = [r for r in results if r['Exp_key']==exp_key]
    exp_res   = sorted(exp_res, key=lambda r: model_order.index(r['Model']))

    for r, ls in zip(exp_res, linestyles):
        fpr, tpr, _ = roc_curve(r['y_te'], r['y_prob'])
        ax.plot(fpr, tpr,
                label=f"{r['Model']}  ({r['ROC-AUC']:.3f})",
                color=MODEL_COLORS[r['Model']], lw=2, linestyle=ls)

    ax.plot([0,1],[0,1],'k--', lw=1, alpha=0.5, label='Random (0.500)')
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate',  fontsize=10)
    ax.set_title(f"ROC Curves — Exp {exp_label}", fontsize=11)
    ax.legend(fontsize=7.5, loc='lower right')
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

# Add legend explanation
axes[0].text(0.02, 0.35,
    "Solid  = original models\nDashed = extended models",
    transform=axes[0].transAxes, fontsize=8,
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('ROC-AUC Curves — All 6 Models × 3 Experimental Conditions',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('plots/combined_item11_roc_all.png', bbox_inches='tight')
plt.close()
print("Saved: plots/combined_item11_roc_all.png")

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATION — Grouped bar: W-F1 all models × all experiments
# ═══════════════════════════════════════════════════════════════════════════════
exp_keys = ['A','B','C']
fig, ax  = plt.subplots(figsize=(14, 6))
n_exp    = 3
n_mod    = len(model_order)
width    = 0.13
x        = np.arange(n_exp)
exp_tick_labels = [EXP_LABELS[k] for k in exp_keys]

for i, mname in enumerate(model_order):
    f1s = [r['W-F1'] for ek in exp_keys
           for r in results if r['Model']==mname and r['Exp_key']==ek]
    ls  = '-' if i < 3 else '--'
    bars = ax.bar(x + i*width, f1s, width, label=mname,
                  color=MODEL_COLORS[mname], alpha=0.85,
                  hatch='' if i<3 else '///')
    for j, v in enumerate(f1s):
        ax.text(x[j]+i*width, v+0.004, f"{v:.3f}",
                ha='center', va='bottom', fontsize=6.5, rotation=90)

ax.set_xticks(x + width*2.5)
ax.set_xticklabels(exp_tick_labels, fontsize=10)
ax.set_ylim(0.45, 0.95)
ax.set_ylabel('Weighted F1-Score', fontsize=11)
ax.set_title('Weighted F1-Score — All 6 Models × 3 Experimental Conditions',
             fontsize=12)
ax.legend(ncol=2, fontsize=9)
ax.axhline(0.70, color='grey', lw=1, linestyle=':', alpha=0.6,
           label='0.70 reference')
plt.tight_layout()
plt.savefig('plots/combined_item9_wf1_all.png', bbox_inches='tight')
plt.close()
print("Saved: plots/combined_item9_wf1_all.png")

# ── AUC Heatmap ──────────────────────────────────────────────────────────────
auc_mat = np.array([[r['ROC-AUC'] for ek in exp_keys
                     for r in results
                     if r['Model']==m and r['Exp_key']==ek]
                    for m in model_order])
fig, ax = plt.subplots(figsize=(8, 5))
im = sns.heatmap(auc_mat, annot=True, fmt='.4f', cmap='YlOrRd',
                 xticklabels=exp_tick_labels, yticklabels=model_order,
                 ax=ax, linewidths=0.5, vmin=0.60, vmax=0.76,
                 annot_kws={'size':10})
ax.set_title('ROC-AUC Heatmap — All Models × Experimental Conditions',
             fontsize=11)
plt.tight_layout()
plt.savefig('plots/combined_item9_auc_heatmap.png', bbox_inches='tight')
plt.close()
print("Saved: plots/combined_item9_auc_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL FILES SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("SAVED MODEL FILES")
print("="*72)
model_files = sorted([f for f in os.listdir('models') if f.endswith('.joblib')])
for f in model_files:
    size_kb = os.path.getsize(f'models/{f}') / 1024
    print(f"  models/{f:<45}  {size_kb:>7.1f} KB")

print(f"\nTotal: {len([f for f in model_files if not f.startswith('scaler')])} "
      f"model files + 3 scaler files")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("FINAL SUMMARY — RANKED BY ROC-AUC")
print("="*72)
df_summary = pd.DataFrame([{
    'Experiment':  r['Experiment'],
    'Model':       r['Model'],
    'Accuracy':    f"{r['Accuracy']*100:.1f}%",
    'W-Precision': round(r['W-Precision'],4),
    'W-Recall':    round(r['W-Recall'],4),
    'W-F1':        round(r['W-F1'],4),
    'ROC-AUC':     round(r['ROC-AUC'],4),
} for r in sorted(results, key=lambda r: r['ROC-AUC'], reverse=True)])
print(df_summary.to_string(index=False))
