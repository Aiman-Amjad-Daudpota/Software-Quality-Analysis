
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)
DATA_PATH = BASE_DIR.parent / 'Data Cleaning' / 'jm1_cleaned.csv'

df = pd.read_csv(DATA_PATH)
print("=" * 70)
print(f"DATASET: shape={df.shape}")
print("=" * 70)

# Feature/target separation
target = 'defects_encoded'
exclude = ['defects', 'defects_encoded']
features = [c for c in df.columns if c not in exclude]
X = df[features]
y = df[target]


#Descriptive stats
print("\n" + "=" * 70)
print("A.1 — DESCRIPTIVE STATISTICS")
print("=" * 70)
desc = df[features].describe().T
desc['skewness'] = df[features].skew()
print(desc[['mean', 'std', 'min', 'max', 'skewness']].round(3))


desc.round(3).to_csv(BASE_DIR / 'descriptive_stats.csv')



# Classwise averages for each predictors
print("\n" + "=" * 70)
print("A.2 — GROUP-WISE MEANS (defective vs clean)")
print("=" * 70)
group_means = df.groupby('defects')[features].mean().T
group_means.columns = ['Clean (mean)', 'Defective (mean)']
group_means['Ratio (Def/Clean)'] = (group_means['Defective (mean)']
                                    / group_means['Clean (mean)']).replace([np.inf, -np.inf], np.nan)
print(group_means.round(3))
group_means.round(3).to_csv(BASE_DIR / 'groupwise_means.csv')


#Calculate and visualize correlations between features and target
print("\n" + "=" * 70)
print("A.3 — CORRELATION HEATMAP")
print("=" * 70)
corr = df[features + [target]].corr()
plt.figure(figsize=(13, 10))
sns.heatmap(corr, cmap='coolwarm', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.4, cbar_kws={"shrink": 0.7})
plt.title('Pearson Correlation — JM1 Features and Target', fontsize=13)
plt.tight_layout()
plt.savefig(PLOTS_DIR / '01_correlation_heatmap.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: plots/01_correlation_heatmap.png")

# Top correlations with target
target_corr = corr[target].drop([target]).abs().sort_values(ascending=False)
print("\nTop |correlation| with target:")
print(target_corr.head(10).round(3))


#Testing for significant differences in feature distributions between defective and clean modules
print("\n" + "=" * 70)
print("A.4 — T-TEST: each feature, defective vs clean")
print("=" * 70)
ttest_rows = []
for col in features:
    g1 = df[df['defects'] == True][col]
    g0 = df[df['defects'] == False][col]
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    ttest_rows.append({'feature': col, 't_stat': t, 'p_value': p,
                       'significant': 'YES' if p < 0.05 else 'no'})
ttest_df = pd.DataFrame(ttest_rows).sort_values('p_value')
print(ttest_df.round(6).to_string(index=False))
ttest_df.round(6).to_csv(BASE_DIR / 'hypothesis_tests.csv', index=False)

sig_count = (ttest_df['p_value'] < 0.05).sum()
print(f"\n>> {sig_count}/{len(features)} features show significant differences (p<0.05)")




key_features = ['loc', 'v(g)', 'ev(g)', 'iv(g)', 'd', 'branchCount']

# Distributions
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, col in zip(axes.flatten(), key_features):
    sns.histplot(np.log1p(df[col]), kde=True, ax=ax, color='steelblue')
    ax.set_title(f'log1p({col})')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '02_distributions.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: plots/02_distributions.png")

# Boxplots split by defect class
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, col in zip(axes.flatten(), key_features):
    sns.boxplot(x='defects', y=col, data=df, ax=ax,
                hue='defects', palette='Set2', legend=False, showfliers=False)
    ax.set_title(f'{col} by defects')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '03_boxplots_by_target.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: plots/03_boxplots_by_target.png")

# Class balance bar
fig, ax = plt.subplots(figsize=(6, 4))
counts = df['defects'].value_counts()
ax.bar(['Clean (False)', 'Defective (True)'], counts.values,
       color=['#5DADE2', '#E74C3C'])
for i, v in enumerate(counts.values):
    ax.text(i, v + 50, str(v), ha='center', fontsize=11)
ax.set_ylabel('Number of modules')
ax.set_title('Class Distribution — JM1 (after cleaning)')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '04_class_balance.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: plots/04_class_balance.png")



# Feature selection
print("\n" + "=" * 70)
print("B — FEATURE SELECTION (4 methods)")
print("=" * 70)

#Mutual Information
mi = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({'feature': features, 'MI': mi}).sort_values('MI', ascending=False)

#ANOVA F-score
f_scores, _ = f_classif(X, y)
f_df = pd.DataFrame({'feature': features, 'F_score': f_scores}).sort_values('F_score', ascending=False)

#Random Forest importance
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X, y)
rf_df = pd.DataFrame({'feature': features, 'RF_imp': rf.feature_importances_}).sort_values('RF_imp', ascending=False)

#Correlation magnitude (|Pearson|)
corr_target = df[features + [target]].corr()[target].drop(target).abs()
c_df = pd.DataFrame({'feature': corr_target.index, 'abs_corr': corr_target.values}).sort_values('abs_corr', ascending=False)

print("\n-- Top 5 by Mutual Information --")
print(mi_df.head().round(4).to_string(index=False))
print("\n-- Top 5 by ANOVA F-score --")
print(f_df.head().round(4).to_string(index=False))
print("\n-- Top 5 by Random Forest importance --")
print(rf_df.head().round(4).to_string(index=False))
print("\n-- Top 5 by |Pearson correlation| --")
print(c_df.head().round(4).to_string(index=False))


#Combined ranking
combined = pd.DataFrame({'feature': features})
combined['rank_MI']   = combined['feature'].map({f: r for r, f in enumerate(mi_df['feature'], 1)})
combined['rank_F']    = combined['feature'].map({f: r for r, f in enumerate(f_df['feature'], 1)})
combined['rank_RF']   = combined['feature'].map({f: r for r, f in enumerate(rf_df['feature'], 1)})
combined['rank_corr'] = combined['feature'].map({f: r for r, f in enumerate(c_df['feature'], 1)})
combined['avg_rank']  = combined[['rank_MI', 'rank_F', 'rank_RF', 'rank_corr']].mean(axis=1)
combined = combined.sort_values('avg_rank').reset_index(drop=True)

print("\n-- Combined ranking (lower avg_rank = stronger feature) --")
print(combined.round(2).to_string(index=False))
combined.to_csv(BASE_DIR / 'feature_ranking.csv', index=False)


#Visualize top 10 features by RF importance
fig, ax = plt.subplots(figsize=(9, 6))
top10 = rf_df.head(10)
sns.barplot(x='RF_imp', y='feature', data=top10,
            hue='feature', palette='viridis', legend=False, ax=ax)
ax.set_title('Top 10 Features by Random Forest Importance')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig(PLOTS_DIR / '05_rf_feature_importance.png', dpi=120, bbox_inches='tight')
plt.close()
print("\nSaved: plots/05_rf_feature_importance.png")


#Using 5 fold CV to check if the top features have predictive signal
print("\n" + "=" * 70)
print("B.5 — SIGNAL SANITY CHECK (Random Forest 5-fold CV)")
print("=" * 70)
rf_cv = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
acc = cross_val_score(rf_cv, X, y, cv=5, scoring='accuracy', n_jobs=-1)
roc = cross_val_score(rf_cv, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
baseline = max(y.mean(), 1 - y.mean())
print(f"Majority-class baseline accuracy: {baseline:.4f}")
print(f"Random Forest accuracy (CV):     {acc.mean():.4f} (+/- {acc.std():.4f})")
print(f"Random Forest ROC-AUC (CV):      {roc.mean():.4f} (+/- {roc.std():.4f})")
print(f"Lift over baseline (accuracy):   {acc.mean() - baseline:+.4f}")

print("\n" + "=" * 70)
print("MILESTONE 2 COMPLETE")
print("=" * 70)
