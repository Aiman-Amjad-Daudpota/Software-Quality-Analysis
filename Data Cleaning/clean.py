
import pandas as pd
import numpy as np


df = pd.read_csv('jm1.csv')
print("=" * 65)
print("ORIGINAL DATASET LOADED")
print("=" * 65)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")
print("Data types:")
print(df.dtypes)
print()


# Converting string columns to numeric, coercing errors to NaN

print("=" * 65)
print("CHECK 1: STRING-TYPED COLUMNS (should be numeric)")
print("=" * 65)

# Per the dataset documentation, ALL features except 'defects' are numeric.
# Check which columns came in as object/string instead of float/int.
expected_numeric = [c for c in df.columns if c != 'defects']
suspect_cols = [c for c in expected_numeric
                if df[c].dtype == 'object' or str(df[c].dtype) == 'str']

if suspect_cols:
    print(f"ISSUE FOUND — columns stored as strings: {suspect_cols}")
    # Show some non-numeric values that explain why
    for col in suspect_cols:
        non_numeric = df[col][pd.to_numeric(df[col], errors='coerce').isna()]
        unique_bad = non_numeric.unique()[:5]
        print(f"  {col}: {len(non_numeric)} non-numeric values "
              f"(samples: {list(unique_bad)})")
    # FIX: coerce to numeric, invalid entries become NaN
    for col in suspect_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print("FIXED — Coerced to numeric; bad values are now NaN.\n")
else:
    print("PASS — All numeric columns are correctly typed.\n")



# Checking for missing values after coercion

print("=" * 65)
print("CHECK 2: MISSING VALUES")
print("=" * 65)
missing = df.isnull().sum()
total_missing = missing.sum()
if total_missing == 0:
    print("PASS — No missing values in any column.\n")
else:
    print(f"ISSUE FOUND — {total_missing} missing cells across:")
    print(missing[missing > 0])
    pct = (total_missing / (df.shape[0] * df.shape[1])) * 100
    print(f"Total missing: {pct:.3f}% of all cells")

    
    rows_with_any_missing = df.isnull().any(axis=1).sum()
    print(f"Rows with at least one missing value: {rows_with_any_missing}")

    #To impute, we fill numeric columns with median
    for col in df.columns:
        if df[col].isnull().any() and df[col].dtype != 'bool':
            df[col] = df[col].fillna(df[col].median())
    print("FIXED — Filled missing values with column median.\n")


#Removing Duplicate Rows
print("=" * 65)
print("CHECK 3: DUPLICATE ROWS")
print("=" * 65)
dupes = df.duplicated().sum()
print(f"Fully duplicated rows: {dupes}")
if dupes > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"FIXED — Removed {dupes} duplicate rows.\n")
else:
    print("PASS — No duplicates to remove.\n")


#Final Check on datatypes after cleaning
print("=" * 65)
print("CHECK 4: FINAL DATA TYPES")
print("=" * 65)
print(df.dtypes)
print()


#Checking class balance of target variable
print("=" * 65)
print("CHECK 5: CLASS BALANCE OF TARGET 'defects'")
print("=" * 65)
counts = df['defects'].value_counts()
pct = df['defects'].value_counts(normalize=True) * 100
print(pd.DataFrame({'count': counts, 'percent': pct.round(2)}))

minority_pct = pct.min()
if minority_pct < 20:
    print(f"\nWARNING — Severe imbalance (minority class = {minority_pct:.1f}%).")
    print("  Use SMOTE or class_weight='balanced' during modeling.")
elif minority_pct < 35:
    print(f"\nNOTE — Mild imbalance (minority class = {minority_pct:.1f}%).")
    print("  Consider class_weight='balanced' for better recall.")
else:
    print("\nPASS — Classes are reasonably balanced.")
print()


#Checking numerical ranges and outliers
print("=" * 65)
print("CHECK 6: NUMERICAL RANGES & OUTLIERS")
print("=" * 65)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
outlier_summary = []
for col in numeric_cols:
    desc = df[col].describe()
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    pct_out = (outliers / len(df)) * 100
    outlier_summary.append({
        'column': col, 'min': desc['min'], 'max': desc['max'],
        'mean': desc['mean'], 'std': desc['std'],
        'outliers': outliers, 'pct_outliers': pct_out
    })

out_df = pd.DataFrame(outlier_summary)
print(out_df.round(3).to_string(index=False))

# Domain sanity checks for software metrics
print("\n-- Domain-specific sanity checks --")
issues = []
if (df['loc'] < 0).any():
    issues.append(f"  {(df['loc'] < 0).sum()} rows with negative LOC")
if 'v(g)' in df.columns and (df['v(g)'] < 0).any():
    issues.append(f"  {(df['v(g)'] < 0).sum()} rows with negative cyclomatic complexity")

if issues:
    print("ISSUES FOUND:")
    for i in issues:
        print(i)
else:
    print("PASS — All metric values are non-negative as expected.")


print("\nDecision: Outliers retained — large/complex modules are")
print("          legitimate data points in defect prediction context.\n")


#adding encoded target for mutual information calculation
print("=" * 65)
print("CHECK 7: ENCODE BINARY TARGET")
print("=" * 65)
df['defects_encoded'] = df['defects'].astype(int)
print("FIXED — Created 'defects_encoded' column (1=defective, 0=clean).")
print(df[['defects', 'defects_encoded']].head().to_string(index=False))
print()


#final Summary of cleaning results
print("=" * 65)
print("FINAL VALIDATION SUMMARY")
print("=" * 65)
print(f"Final shape:       {df.shape}")
print(f"Missing values:    {df.isnull().sum().sum()}")
print(f"Duplicate rows:    {df.duplicated().sum()}")
print(f"Numeric features:  {len([c for c in df.columns if df[c].dtype != 'bool' and c != 'defects'])}")
print(f"Target classes:    {df['defects'].nunique()}")
print()

output_path = 'jm1_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to: {output_path}")
print("=" * 65)
print("CLEANING COMPLETE — Dataset ready for EDA & modeling.")
print("=" * 65)
