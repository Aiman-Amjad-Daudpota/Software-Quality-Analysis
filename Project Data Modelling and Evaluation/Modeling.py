import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay,
                             precision_recall_curve, average_precision_score)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
os.makedirs('plots', exist_ok=True)


# LOAD DATA & DEFINE FEATURES# 
df = pd.read_csv('../Data Cleaning/jm1_cleaned.csv')

# Top features from Milestone 2 combined ranking
TOP_FEATURES = ['loc', 'lOCode', 'uniq_Opnd', 'total_Op',
                'v', 'v(g)', 'lOBlank', 'n', 'branchCount', 'total_Opnd']
ALL_FEATURES = [c for c in df.columns
                if c not in ('defects', 'defects_encoded')]

X_top  = df[TOP_FEATURES]
X_all  = df[ALL_FEATURES]
y      = df['defects_encoded']


print(f"Dataset shape : {df.shape}")
print(f"Top features  : {TOP_FEATURES}")
print(f"Class balance : {y.value_counts().to_dict()}")



# Classification modeling to predict defect proneness
# Builds and evaluates models to predict defect proneness

print("CLASSIFICATION MODELS")



# Train / Test split  (stratified 80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_top, y, test_size=0.20, random_state=42, stratify=y
)


# Handle class imbalance with SMOTE on training set only
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"\nTraining set before SMOTE : {dict(pd.Series(y_train).value_counts())}")
print(f"Training set after  SMOTE : {dict(pd.Series(y_train_sm).value_counts())}")
print(f"Test set (unchanged)      : {dict(pd.Series(y_test).value_counts())}")



# Define three classifiers with basic hyperparameters:
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
                       max_iter=1000, random_state=42,
                       class_weight="balanced"))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=20,
        random_state=42, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=15,
        min_samples_leaf=5,  random_state=42,
        class_weight="balanced", n_jobs=-1
    ),
}



# Performing 5-Fold Stratified Cross-Validation on SMOTE'd training set

print("5-FOLD CROSS-VALIDATION RESULTS (training set)")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_summary = []

for name, model in models.items():
    scores = cross_validate(
        model, X_train_sm, y_train_sm, cv=cv,
        scoring=["accuracy", "f1", "roc_auc"],
        return_train_score=False, n_jobs=-1
    )
    row = {
        "Model":    name,
        "CV Acc":   f"{scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}",
        "CV F1":    f"{scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}",
        "CV AUC":   f"{scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}",
    }
    cv_summary.append(row)
    print(f"  {name:25s}  Acc={scores['test_accuracy'].mean():.4f}  "
          f"F1={scores['test_f1'].mean():.4f}  "
          f"AUC={scores['test_roc_auc'].mean():.4f}")

cv_df = pd.DataFrame(cv_summary)



# Train final models on full SMOTE training set and evaluating on the test set
print("Traing and Test set Evaluation")
test_summary = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
    trained_models[name] = model

    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]

    report  = classification_report(y_test, y_pred, output_dict=True)
    auc     = roc_auc_score(y_test, y_prob)

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred,
          target_names=["Clean", "Defective"]))
    print(f"  ROC-AUC: {auc:.4f}")

    test_summary.append({
        "Model":        name,
        "Accuracy":     report["accuracy"],
        "Precision":    report["weighted avg"]["precision"],
        "Recall":       report["weighted avg"]["recall"],
        "F1 (weighted)":report["weighted avg"]["f1-score"],
        "ROC-AUC":      auc,
    })

test_df = pd.DataFrame(test_summary)
print("\nSummary Table")
print(test_df.round(4).to_string(index=False))
test_df.to_csv("model_evaluation_results.csv", index=False)



# Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, model) in zip(axes, trained_models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Clean", "Defective"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=11)
plt.tight_layout()
plt.savefig("plots/06_confusion_matrices.png", dpi=120, bbox_inches='tight')
plt.close()




#ROC Curves (all three on one plot)
fig, ax = plt.subplots(figsize=(7, 6))
colors = ['#2196F3', '#FF9800', '#4CAF50']
for (name, model), color in zip(trained_models.items(), colors):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, lw=2)
ax.plot([0,1],[0,1],'k--', lw=1, label="Random baseline (AUC = 0.500)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Three Classifiers")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plots/07_roc_curves.png", dpi=120, bbox_inches='tight')
plt.close()




#Precision-Recall Curves
fig, ax = plt.subplots(figsize=(7, 6))
for (name, model), color in zip(trained_models.items(), colors):
    y_prob = model.predict_proba(X_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax.plot(rec, prec, label=f"{name} (AP = {ap:.3f})", color=color, lw=2)
baseline = y_test.mean()
ax.axhline(baseline, color='k', linestyle='--', lw=1,
           label=f"Random baseline (AP = {baseline:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — All Three Classifiers")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("plots/08_precision_recall_curves.png", dpi=120, bbox_inches='tight')
plt.close()




# Decision Tree visualization (shallow)
fig, ax = plt.subplots(figsize=(18, 7))
plot_tree(
    trained_models["Decision Tree"],
    feature_names=TOP_FEATURES,
    class_names=["Clean", "Defective"],
    max_depth=3,                  # show top 3 levels only for readability
    filled=True, rounded=True,
    fontsize=9, ax=ax
)
ax.set_title("Decision Tree — Top 3 Levels (full depth = 8)")
plt.tight_layout()
plt.savefig("plots/09_decision_tree.png", dpi=120, bbox_inches='tight')
plt.close()




#FEATURE IMPORTANCE ANALYSIS
# Explains WHICH metrics most strongly indicate defect proneness
print("\n" + "#" * 70)
print("# SECTION B: FEATURE IMPORTANCE — OBJECTIVE 1")
print("#  'Analyze which metrics most strongly indicate defects'")
print("#" * 70)

# ----Logistic Regression coefficients ----
lr_clf  = trained_models["Logistic Regression"].named_steps["clf"]
lr_coef = np.abs(lr_clf.coef_[0])
lr_df   = pd.DataFrame({"feature": TOP_FEATURES, "abs_coef": lr_coef})
lr_df   = lr_df.sort_values("abs_coef", ascending=False).reset_index(drop=True)

# ---- Decision Tree feature importance ----
dt_clf = trained_models["Decision Tree"]
dt_df  = pd.DataFrame({"feature": TOP_FEATURES,
                        "importance": dt_clf.feature_importances_})
dt_df  = dt_df.sort_values("importance", ascending=False).reset_index(drop=True)

# ----Random Forest feature importance ----
rf_clf = trained_models["Random Forest"]
rf_df  = pd.DataFrame({"feature": TOP_FEATURES,
                        "importance": rf_clf.feature_importances_})
rf_df  = rf_df.sort_values("importance", ascending=False).reset_index(drop=True)

# ---- Combined ranking ----
fi_combined = pd.DataFrame({"feature": TOP_FEATURES})
fi_combined["rank_LR"] = fi_combined["feature"].map(
    {f: r for r, f in enumerate(lr_df["feature"], 1)})
fi_combined["rank_DT"] = fi_combined["feature"].map(
    {f: r for r, f in enumerate(dt_df["feature"], 1)})
fi_combined["rank_RF"] = fi_combined["feature"].map(
    {f: r for r, f in enumerate(rf_df["feature"], 1)})
fi_combined["avg_rank"] = fi_combined[["rank_LR","rank_DT","rank_RF"]].mean(axis=1)
fi_combined = fi_combined.sort_values("avg_rank").reset_index(drop=True)

print("\nCombined feature importance from all 3 classifiers:")
print(fi_combined.round(2).to_string(index=False))
fi_combined.to_csv("feature_importance_modeling.csv", index=False)

# ---- Plot: side-by-side importance ----
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, (df_fi, title, col) in zip(
    axes,
    [(lr_df, "Logistic Regression\n(|coefficient|)", "abs_coef"),
     (dt_df, "Decision Tree\n(Gini importance)", "importance"),
     (rf_df, "Random Forest\n(mean decrease impurity)", "importance")]
):
    sns.barplot(x=col, y="feature", data=df_fi,
                hue="feature", palette="viridis", legend=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
plt.suptitle("Feature Importance — Objective 1: Which Metrics Predict Defects?",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("plots/10_feature_importance_comparison.png",
            dpi=120, bbox_inches='tight')
plt.close()


# K-MEANS CLUSTERING
print("# K-MEANS CLUSTERING ")
print("#  'Group modules into risk profiles based on metrics'")

# Use top 5 features for clustering 
CLUSTER_FEATURES = ['loc', 'v(g)', 'branchCount', 'uniq_Opnd', 'lOCode']
X_clust = df[CLUSTER_FEATURES].copy()


scaler_c = StandardScaler()
X_scaled = scaler_c.fit_transform(X_clust)

# ----Elbow Method ----
inertias, sil_scores = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(K_range), inertias, 'bo-', lw=2)
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (within-cluster sum of squares)")
axes[0].set_title("Elbow Method — Finding Optimal k")
axes[1].plot(list(K_range), sil_scores, 'ro-', lw=2)
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score — Finding Optimal k")
plt.tight_layout()
plt.savefig("plots/11_elbow_silhouette.png", dpi=120, bbox_inches='tight')
plt.close()

best_k_sil = list(K_range)[np.argmax(sil_scores)]

best_k = 3
print(f"\nElbow/Silhouette analysis complete.")
print(f"Mathematically optimal k (silhouette): {best_k_sil}")
print(f"Selected k=3 for interpretable Low/Medium/High risk profiles.")


# Fit final k-means with k=3
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster"] = km_final.fit_predict(X_scaled)

#Profile each cluster
print(f"\nCluster profiles (k={best_k}):")
profile = df.groupby("cluster")[CLUSTER_FEATURES + ["defects_encoded"]].mean()
profile["size"] = df.groupby("cluster")["defects_encoded"].count()
profile["defect_rate_%"] = (profile["defects_encoded"] * 100).round(1)
print(profile.round(2))

#Data-driven naming
dataset_avg_defect = df["defects_encoded"].mean() * 100  # 22.5%

def name_cluster(row):
    loc   = row["loc"]
    rate  = row["defect_rate_%"]

    # Complexity tier based on average LOC
    if loc < 100:
        complexity = "Small & Simple"
    elif loc < 500:
        complexity = "Moderate Complexity"
    else:
        complexity = "Large & Critical"

    # Defect qualifier relative to dataset average (22.5%)
    if rate <= dataset_avg_defect:             # at or below average
        qualifier = "(Near-Average Defect Rate)"
    elif rate < dataset_avg_defect * 2.5:      # moderately above average
        qualifier = "(Elevated Defect Rate)"
    else:                                       # far above average
        qualifier = "(Very High Defect Rate)"

    return f"{complexity} {qualifier}"

cluster_names = {}
for cid, row in profile.iterrows():
    cluster_names[cid] = name_cluster(row)

df["cluster_name"] = df["cluster"].map(cluster_names)
profile["cluster_name"] = profile.index.map(cluster_names)
profile.to_csv("cluster_profiles.csv")

print("\nData-driven cluster naming:")
print(f"  (Dataset average defect rate: {dataset_avg_defect:.1f}%)")
for cid, name in cluster_names.items():
    rate = profile.loc[cid, "defect_rate_%"]
    size = int(profile.loc[cid, "size"])
    avg_loc = profile.loc[cid, "loc"]
    avg_vg  = profile.loc[cid, "v(g)"]
    print(f"\n  Cluster {cid}: '{name}'")
    print(f"    avg LOC={avg_loc:.0f}, avg v(g)={avg_vg:.1f}, "
          f"defect rate={rate}%, n={size}")


short_labels = {}
sorted_by_loc = profile["loc"].sort_values()
tier_names = ["Tier 1 — Small & Simple",
              "Tier 2 — Moderate Complexity",
              "Tier 3 — Large & Critical"]
for i, cid in enumerate(sorted_by_loc.index):
    short_labels[cid] = tier_names[i]

df["tier_label"] = df["cluster"].map(short_labels)

# Visualize clusters 
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

palette = {
    "Tier 1 — Small & Simple":       "#2ecc71",
    "Tier 2 — Moderate Complexity":  "#f39c12",
    "Tier 3 — Large & Critical":     "#e74c3c",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: coloured by tier label
for label, color in palette.items():
    mask = df["tier_label"] == label
    if mask.any():
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=color, label=label, alpha=0.4, s=8)
axes[0].set_title("K-Means Clusters (k=3) — PCA 2D Projection")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
axes[0].legend(fontsize=8)

# Right: actual defect labels
colors_actual = ['#3498db' if v == 0 else '#e74c3c'
                 for v in df["defects_encoded"]]
axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=colors_actual, alpha=0.3, s=8)
p1 = mpatches.Patch(color='#3498db', label='Clean (actual)')
p2 = mpatches.Patch(color='#e74c3c', label='Defective (actual)')
axes[1].legend(handles=[p1, p2])
axes[1].set_title("Actual Defect Labels — PCA 2D Projection")
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")

plt.tight_layout()
plt.savefig("plots/12_cluster_visualization.png", dpi=120, bbox_inches='tight')
plt.close()


#Cluster profile bar charts
profile_sorted = profile.sort_values("loc")
tier_order     = [short_labels[cid] for cid in profile_sorted.index]
defect_rates   = profile_sorted["defect_rate_%"].values
cluster_sizes  = profile_sorted["size"].values
bar_colors     = [palette.get(t, '#999') for t in tier_order]
short_labels_display = [t.split(" — ")[1] for t in tier_order]  # "Small & Simple" etc.

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(short_labels_display, defect_rates, color=bar_colors)
axes[0].set_ylabel("Defect Rate (%)")
axes[0].set_title("Defect Rate per Cluster Tier")
for i, v in enumerate(defect_rates):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=10)

axes[1].bar(short_labels_display, cluster_sizes, color=bar_colors)
axes[1].set_ylabel("Number of Modules")
axes[1].set_title("Module Count per Cluster Tier")
for i, v in enumerate(cluster_sizes):
    axes[1].text(i, v + 30, str(int(v)), ha='center', fontsize=10)

plt.suptitle("K-Means Cluster Summary — Objective 3 (k=3)", fontsize=12)
plt.tight_layout()
plt.savefig("plots/13_cluster_profiles.png", dpi=120, bbox_inches='tight')
plt.close()