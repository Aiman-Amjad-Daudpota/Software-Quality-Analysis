import os, warnings, joblib, pandas as pd, numpy as np
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, ConfusionMatrixDisplay,
                             roc_curve, precision_recall_curve, average_precision_score)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_style("whitegrid")
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)   # <-- folder for saved models

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
df = pd.read_csv("../Data Cleaning/jm1_cleaned.csv")

TOP_FEATURES    = ['loc', 'lOCode', 'uniq_Opnd', 'total_Op',
                   'v', 'v(g)', 'lOBlank', 'n', 'branchCount', 'total_Opnd']
CLUSTER_FEATURES= ['loc', 'v(g)', 'branchCount', 'uniq_Opnd', 'lOCode']
ALL_FEATURES    = [c for c in df.columns if c not in ('defects', 'defects_encoded')]

X_top = df[TOP_FEATURES]
y     = df['defects_encoded']

print(f"Dataset shape : {df.shape}")
print(f"Top features  : {TOP_FEATURES}")
print(f"Class balance : {y.value_counts().to_dict()}")

# ── TRAIN / TEST SPLIT ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_top, y, test_size=0.20, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"\nTraining set before SMOTE : {dict(pd.Series(y_train).value_counts())}")
print(f"Training set after  SMOTE : {dict(pd.Series(y_train_sm).value_counts())}")
print(f"Test set (unchanged)      : {dict(pd.Series(y_test).value_counts())}")

# ── CLASSIFIERS ────────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=20, random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        random_state=42, class_weight="balanced", n_jobs=-1),
}

# ── 5-FOLD CV ─────────────────────────────────────────────────────────────────
print("\n── 5-FOLD CROSS-VALIDATION (training set) ──")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for name, model in models.items():
    scores = cross_validate(model, X_train_sm, y_train_sm, cv=cv,
                            scoring=["accuracy","f1","roc_auc"],
                            return_train_score=False, n_jobs=-1)
    cv_rows.append({"Model": name,
                    "CV Acc": f"{scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}",
                    "CV F1":  f"{scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}",
                    "CV AUC": f"{scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}"})
    print(f"  {name:25s}  Acc={scores['test_accuracy'].mean():.4f}  "
          f"F1={scores['test_f1'].mean():.4f}  AUC={scores['test_roc_auc'].mean():.4f}")

# ── TRAIN FINAL + EVALUATE ON TEST ────────────────────────────────────────────
print("\n── TEST SET EVALUATION ──")
test_rows, trained_models = [], {}
for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    trained_models[name] = model

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    auc    = roc_auc_score(y_test, y_prob)

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["Clean","Defective"]))
    print(f"  ROC-AUC: {auc:.4f}")

    test_rows.append({"Model": name,
                      "Accuracy":      report["accuracy"],
                      "Precision":     report["weighted avg"]["precision"],
                      "Recall":        report["weighted avg"]["recall"],
                      "F1 (weighted)": report["weighted avg"]["f1-score"],
                      "ROC-AUC":       auc})

pd.DataFrame(test_rows).round(4).to_csv("model_evaluation_results.csv", index=False)

# ── SAVE CLASSIFICATION MODELS  ← THE FIX ─────────────────────────────────────
print("\n── SAVING CLASSIFICATION MODELS ──")
for name, model in trained_models.items():
    filename = "models/" + name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, filename)
    print(f"  Saved: {filename}")

# also save the feature list so the predict script knows the column order
joblib.dump(TOP_FEATURES, "models/top_features.pkl")
print("  Saved: models/top_features.pkl")

# ── PLOTS ─────────────────────────────────────────────────────────────────────
colors = ['#2196F3','#FF9800','#4CAF50']

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, model) in zip(axes, trained_models.items()):
    cm = confusion_matrix(y_test, model.predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=["Clean","Defective"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)
plt.tight_layout()
plt.savefig("plots/06_confusion_matrices.png", dpi=120, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(7,6))
for (name, model), color in zip(trained_models.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)
ax.plot([0,1],[0,1],'k--', lw=1)
ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curves")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plots/07_roc_curves.png", dpi=120, bbox_inches='tight')
plt.close()
print("Saved: plots/06 & 07")

# ── K-MEANS CLUSTERING ────────────────────────────────────────────────────────
print("\n── K-MEANS CLUSTERING ──")
X_clust = df[CLUSTER_FEATURES].copy()
scaler_c = StandardScaler()
X_scaled = scaler_c.fit_transform(X_clust)

inertias, sil_scores = [], []
for k in range(2,9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

best_k = 3
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster"] = km_final.fit_predict(X_scaled)

profile = df.groupby("cluster")[CLUSTER_FEATURES + ["defects_encoded"]].mean()
profile["size"] = df.groupby("cluster")["defects_encoded"].count()
profile["defect_rate_%"] = (profile["defects_encoded"] * 100).round(1)
print(profile.round(2))

sorted_by_loc = profile["loc"].sort_values()
tier_map = {}
tier_names = ["Tier 1 — Small & Simple",
              "Tier 2 — Moderate Complexity",
              "Tier 3 — Large & Critical"]
for i, cid in enumerate(sorted_by_loc.index):
    tier_map[cid] = tier_names[i]
df["tier_label"] = df["cluster"].map(tier_map)

profile["tier_label"] = profile.index.map(tier_map)
profile.to_csv("cluster_profiles.csv")

# ── SAVE CLUSTERING MODEL + SCALER  ← THE FIX ────────────────────────────────
print("\n── SAVING CLUSTERING MODELS ──")
joblib.dump(km_final,         "models/kmeans_k3.pkl")
joblib.dump(scaler_c,         "models/cluster_scaler.pkl")
joblib.dump(CLUSTER_FEATURES, "models/cluster_features.pkl")
joblib.dump(tier_map,         "models/tier_map.pkl")
print("  Saved: models/kmeans_k3.pkl")
print("  Saved: models/cluster_scaler.pkl")
print("  Saved: models/cluster_features.pkl")
print("  Saved: models/tier_map.pkl")

print("\n── ALL MODELS SAVED ──")
print("Files in models/:", os.listdir("models/"))
