"""
predict.py — Interactive terminal predictor for JM1 defect models
Usage:
    python predict.py              (interactive mode — prompts for each metric)
    python predict.py --demo       (runs a quick demo with two sample modules)
"""

import sys, os, joblib, numpy as np, pandas as pd

# ── Load saved models ─────────────────────────────────────────────────────────
MODEL_DIR = os.path.dirname(__file__)

def load(name):
    return joblib.load(os.path.join(MODEL_DIR, name))

lr_model        = load("logistic_regression.pkl")
dt_model        = load("decision_tree.pkl")
rf_model        = load("random_forest.pkl")
top_features    = load("top_features.pkl")

km_model        = load("kmeans_k3.pkl")
cluster_scaler  = load("cluster_scaler.pkl")
cluster_features= load("cluster_features.pkl")
tier_map        = load("tier_map.pkl")

# ── Helpers ───────────────────────────────────────────────────────────────────
SEP  = "=" * 60
SEP2 = "-" * 60

FEATURE_DESCRIPTIONS = {
    "loc":        "Lines of Code (total, e.g. 150)",
    "lOCode":     "Lines of executable code (e.g. 80)",
    "uniq_Opnd":  "Unique operands — distinct variables/literals (e.g. 25)",
    "total_Op":   "Total operators used (e.g. 60)",
    "v":          "Halstead Volume (e.g. 500.0)",
    "v(g)":       "Cyclomatic Complexity (e.g. 5)",
    "lOBlank":    "Blank lines (e.g. 20)",
    "n":          "Halstead Program Length n1+n2 (e.g. 120)",
    "branchCount":"Number of branches/decision points (e.g. 8)",
    "total_Opnd": "Total operands used (e.g. 70)",
}

CLUSTER_FEATURE_DESCRIPTIONS = {
    "loc":        "Lines of Code",
    "v(g)":       "Cyclomatic Complexity",
    "branchCount":"Branch Count",
    "uniq_Opnd":  "Unique Operands",
    "lOCode":     "Lines of Executable Code",
}


def prompt_float(feature, description, default=None):
    """Prompt the user for a single float value."""
    hint = f"  [default={default}]" if default is not None else ""
    while True:
        raw = input(f"  {feature:15s} ({description}){hint}: ").strip()
        if raw == "" and default is not None:
            return float(default)
        try:
            val = float(raw)
            if val < 0:
                print("    ⚠  Value must be ≥ 0. Try again.")
                continue
            return val
        except ValueError:
            print("    ⚠  Please enter a number.")


def classify(values_dict):
    """Run all three classifiers and print results."""
    X = pd.DataFrame([values_dict])[top_features]

    print(f"\n{SEP2}")
    print("  CLASSIFICATION RESULTS")
    print(SEP2)

    probs, preds = {}, {}
    for name, model in [("Logistic Regression", lr_model),
                         ("Decision Tree",       dt_model),
                         ("Random Forest",       rf_model)]:
        prob = model.predict_proba(X)[0][1]
        pred = model.predict(X)[0]
        probs[name] = prob
        preds[name] = pred
        verdict = "⚠  DEFECTIVE" if pred == 1 else "✓  CLEAN"
        print(f"  {name:25s}  →  {verdict}   (defect prob: {prob:.1%})")

    # Majority vote
    votes = sum(preds.values())
    final = "⚠  DEFECTIVE" if votes >= 2 else "✓  CLEAN"
    avg_prob = np.mean(list(probs.values()))
    print(f"\n  {'Majority Vote (2-of-3)':25s}  →  {final}   (avg prob: {avg_prob:.1%})")
    print(SEP2)


def cluster(values_dict):
    """Assign module to a risk cluster and print results."""
    X_c = pd.DataFrame([values_dict])[cluster_features]
    X_scaled = cluster_scaler.transform(X_c)
    cluster_id = km_model.predict(X_scaled)[0]
    tier = tier_map.get(cluster_id, f"Cluster {cluster_id}")

    # Compute distances to all centroids for context
    dists = km_model.transform(X_scaled)[0]
    dist_info = ", ".join(
        [f"Tier {i+1}={d:.2f}" for i, d in enumerate(
            [dists[k] for k in sorted(tier_map.keys())])])

    print(f"\n{SEP2}")
    print("  CLUSTERING RESULT")
    print(SEP2)
    print(f"  Assigned Cluster  →  {tier}")
    print(f"  Centroid distances: ({dist_info})")

    risk_level = {
        "Tier 1 — Small & Simple":      "🟢 LOW RISK",
        "Tier 2 — Moderate Complexity":  "🟡 MEDIUM RISK",
        "Tier 3 — Large & Critical":     "🔴 HIGH RISK",
    }.get(tier, "❓ UNKNOWN")
    print(f"  Risk Level        →  {risk_level}")
    print(SEP2)


def interactive_mode():
    print(SEP)
    print("  JM1 SOFTWARE DEFECT PREDICTOR")
    print("  Enter software metrics for a module to predict defect risk")
    print(SEP)

    while True:
        print("\nEnter module metrics (press Enter to accept default values):\n")

        # Classification features
        clf_values = {}
        for feat in top_features:
            desc = FEATURE_DESCRIPTIONS.get(feat, feat)
            clf_values[feat] = prompt_float(feat, desc, default=1.0)

        # For clustering, some features overlap — reuse where possible
        clust_values = {}
        print(f"\n{SEP2}")
        print("  A few extra metrics are needed for cluster assignment:")
        print(SEP2)
        for feat in cluster_features:
            if feat in clf_values:
                clust_values[feat] = clf_values[feat]
            else:
                desc = CLUSTER_FEATURE_DESCRIPTIONS.get(feat, feat)
                clust_values[feat] = prompt_float(feat, desc, default=1.0)

        # Run predictions
        classify(clf_values)
        cluster(clust_values)

        print(f"\n{SEP}")
        again = input("  Predict another module? [y/N]: ").strip().lower()
        if again != 'y':
            print("\n  Goodbye!\n")
            break


def demo_mode():
    """Run two pre-defined examples to show the system works."""
    print(SEP)
    print("  DEMO MODE — 2 sample modules")
    print(SEP)

    samples = [
        {
            "label": "Sample A — Small, simple module (likely CLEAN)",
            "clf":   {"loc":15,"lOCode":10,"uniq_Opnd":8,"total_Op":12,
                      "v":80.0,"v(g)":2,"lOBlank":3,"n":20,"branchCount":2,"total_Opnd":15},
            "clust": {"loc":15,"v(g)":2,"branchCount":2,"uniq_Opnd":8,"lOCode":10},
        },
        {
            "label": "Sample B — Large, complex module (likely DEFECTIVE)",
            "clf":   {"loc":900,"lOCode":650,"uniq_Opnd":120,"total_Op":500,
                      "v":8000.0,"v(g)":45,"lOBlank":80,"n":800,"branchCount":70,"total_Opnd":600},
            "clust": {"loc":900,"v(g)":45,"branchCount":70,"uniq_Opnd":120,"lOCode":650},
        },
    ]

    for s in samples:
        print(f"\n{'='*60}")
        print(f"  {s['label']}")
        print(f"  Metrics: {s['clf']}")
        classify(s["clf"])
        cluster(s["clust"])

    print(f"\n{SEP}")
    print("  Demo complete. Run without --demo for interactive mode.")
    print(SEP)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo_mode()
    else:
        interactive_mode()
