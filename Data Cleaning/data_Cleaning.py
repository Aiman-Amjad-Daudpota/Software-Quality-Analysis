"""
================================================================
DATASET CLEANING & VALIDATION SCRIPT
Dataset: Code Smells and Refactoring Practices (5000 rows)
================================================================
This script runs DIAGNOSTIC checks first, then applies FIXES
where needed. Each section prints its findings so you can see
exactly what was wrong and what was changed.
"""

import pandas as pd
import numpy as np

# ----------------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------------
df = pd.read_csv('code_smells_refactoring_5000.csv')
print("=" * 65)
print("ORIGINAL DATASET LOADED")
print("=" * 65)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")


# ================================================================
# CHECK 1: MISSING VALUES
# ================================================================
print("=" * 65)
print("CHECK 1: MISSING VALUES")
print("=" * 65)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("PASS — No missing values in any column.\n")
else:
    print("ISSUE FOUND — Missing values detected:")
    print(missing[missing > 0])
    # FIX: fill numerical with median, categorical with mode
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    print("FIXED — Numerical filled with median, categorical with mode.\n")


# ================================================================
# CHECK 2: DUPLICATE ROWS
# ================================================================
print("=" * 65)
print("CHECK 2: DUPLICATE ROWS")
print("=" * 65)
full_dupes = df.duplicated().sum()
print(f"Fully duplicated rows: {full_dupes}")

# Check duplicates on project_id + module_name (logical duplicates)
logical_dupes = df.duplicated(subset=['project_id', 'module_name']).sum()
print(f"Duplicate (project_id + module_name) pairs: {logical_dupes}")

if full_dupes > 0:
    df = df.drop_duplicates()
    print(f"FIXED — Removed {full_dupes} duplicate rows.\n")
else:
    print("PASS — No duplicates to remove.\n")


# ================================================================
# CHECK 3: DATA TYPES
# ================================================================
print("=" * 65)
print("CHECK 3: DATA TYPES")
print("=" * 65)
print(df.dtypes)
print()

# Convert review_date to datetime
df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
invalid_dates = df['review_date'].isnull().sum()
if invalid_dates > 0:
    print(f"WARNING — {invalid_dates} dates could not be parsed.\n")
else:
    print("FIXED — 'review_date' converted to datetime successfully.\n")


# ================================================================
# CHECK 4: CATEGORICAL VALUE CONSISTENCY
# ================================================================
print("=" * 65)
print("CHECK 4: CATEGORICAL VALUE CONSISTENCY (typos/case issues)")
print("=" * 65)
categorical_cols = ['programming_language', 'code_smell_type',
                    'severity_level', 'refactoring_applied',
                    'refactoring_type']

for col in categorical_cols:
    unique_vals = df[col].unique()
    print(f"\n{col}  ({len(unique_vals)} unique values):")
    print(f"  {sorted([str(v) for v in unique_vals])}")

# FIX: strip whitespace and standardize casing to prevent hidden duplicates
for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip()

print("\nFIXED — Stripped whitespace from all categorical columns.\n")


# ================================================================
# CHECK 5: CLASS BALANCE (critical for classification)
# ================================================================
print("=" * 65)
print("CHECK 5: CLASS BALANCE OF TARGET VARIABLES")
print("=" * 65)

print("\n-- refactoring_applied (binary target) --")
ra_counts = df['refactoring_applied'].value_counts()
ra_pct = df['refactoring_applied'].value_counts(normalize=True) * 100
print(pd.DataFrame({'count': ra_counts, 'percent': ra_pct.round(2)}))

# Interpretation
minority_pct = ra_pct.min()
if minority_pct < 20:
    print(f"WARNING — Severe imbalance (minority class = {minority_pct:.1f}%).")
    print("  Use SMOTE or class_weight='balanced' during modeling.")
elif minority_pct < 35:
    print(f"NOTE — Mild imbalance (minority class = {minority_pct:.1f}%).")
    print("  Consider class_weight='balanced' for better recall.")
else:
    print("PASS — Classes are reasonably balanced.")

print("\n-- refactoring_type (multi-class target) --")
rt_counts = df['refactoring_type'].value_counts()
print(rt_counts)
rare_classes = rt_counts[rt_counts < 50]
if len(rare_classes) > 0:
    print(f"\nWARNING — {len(rare_classes)} class(es) have < 50 samples:")
    print(rare_classes)
else:
    print("\nPASS — All refactoring types have sufficient samples.")
print()


# ================================================================
# CHECK 6: NUMERICAL RANGES & OUTLIERS (IQR method)
# ================================================================
print("=" * 65)
print("CHECK 6: NUMERICAL RANGES & OUTLIERS")
print("=" * 65)
numerical_cols = ['lines_of_code', 'maintainability_score', 'bug_density']

for col in numerical_cols:
    desc = df[col].describe()
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
    pct = (outliers / len(df)) * 100

    print(f"\n{col}:")
    print(f"  min={desc['min']:.2f}, max={desc['max']:.2f}, "
          f"mean={desc['mean']:.2f}, std={desc['std']:.2f}")
    print(f"  IQR bounds: [{lower:.2f}, {upper:.2f}]")
    print(f"  Outliers: {outliers} rows ({pct:.2f}%)")

# Sanity range checks specific to this dataset's domain
print("\n-- Domain-specific sanity checks --")
issues = []
if (df['lines_of_code'] <= 0).any():
    issues.append(f"  {(df['lines_of_code'] <= 0).sum()} rows with LOC <= 0")
if (df['maintainability_score'] < 0).any() or (df['maintainability_score'] > 10).any():
    issues.append("  maintainability_score outside expected 0-10 range")
if (df['bug_density'] < 0).any():
    issues.append(f"  {(df['bug_density'] < 0).sum()} rows with negative bug_density")

if issues:
    print("ISSUES FOUND:")
    for i in issues:
        print(i)
else:
    print("PASS — All numerical values fall within expected domain ranges.")

# NOTE: We are NOT removing outliers automatically.
# In a real code-quality dataset, large modules (high LOC) are legitimate
# data points, not errors. Removing them would throw away the most
# "interesting" samples. Document them and proceed.
print("\nDecision: Keeping outliers — they represent legitimate edge cases.")
print("          (Will revisit during modeling if they hurt performance.)\n")


# ================================================================
# CHECK 7: TARGET ENCODING (Yes/No -> 1/0)
# ================================================================
print("=" * 65)
print("CHECK 7: ENCODE BINARY TARGET")
print("=" * 65)
df['refactoring_applied_encoded'] = df['refactoring_applied'].map(
    {'Yes': 1, 'No': 0}
)
print("FIXED — Created 'refactoring_applied_encoded' (1=Yes, 0=No).\n")


# ================================================================
# FINAL VALIDATION SUMMARY
# ================================================================
print("=" * 65)
print("FINAL VALIDATION SUMMARY")
print("=" * 65)
print(f"Final shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Data types:\n{df.dtypes}\n")

# Save the cleaned dataset
output_path = 'cleaned_code_smells_refactoring_5000.csv'
df.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to: {output_path}")
print("=" * 65)
print("CLEANING COMPLETE — Dataset ready for EDA & modeling.")
print("=" * 65)