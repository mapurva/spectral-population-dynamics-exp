import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# Create output folders
os.makedirs("data/analysis", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Load data
df = pd.read_csv("data/results_with_resistance.csv")

# Clean
df = df.dropna()
df = df[(df["mean_fixation"] > 0) & (df["lambda2"] > 0) & (df["kirchhoff"] > 0)]

# ----------------------------
# Regression with CI
# ----------------------------
def regression_with_ci(x, y):
    x_log = np.log(x)
    y_log = np.log(y)

    slope, intercept, r, p, stderr = linregress(x_log, y_log)

    alpha = -slope
    ci_low = alpha - 1.96 * stderr
    ci_high = alpha + 1.96 * stderr

    return alpha, r**2, p, ci_low, ci_high


# ----------------------------
# STORE RESULTS
# ----------------------------
records = []

# Global models
for name, x in {
    "lambda2": df["lambda2"],
    "kirchhoff": df["kirchhoff"]
}.items():

    alpha, r2, p, ci_low, ci_high = regression_with_ci(x, df["mean_fixation"])

    records.append({
        "graph": "GLOBAL",
        "model": name,
        "alpha": alpha,
        "r2": r2,
        "p_value": p,
        "ci_low": ci_low,
        "ci_high": ci_high
    })

# Per graph models
for g in df["graph"].unique():
    sub = df[df["graph"] == g]

    if len(sub) < 3:
        continue

    for name, x in {
        "lambda2": sub["lambda2"],
        "kirchhoff": sub["kirchhoff"]
    }.items():

        alpha, r2, p, ci_low, ci_high = regression_with_ci(x, sub["mean_fixation"])

        records.append({
            "graph": g,
            "model": name,
            "alpha": alpha,
            "r2": r2,
            "p_value": p,
            "ci_low": ci_low,
            "ci_high": ci_high
        })

# Save results
results_df = pd.DataFrame(records)
results_df.to_csv("data/analysis/analysis_results.csv", index=False)

print("Saved: data/analysis/analysis_results.csv")

# ----------------------------
# PLOTTING (PAPER READY)
# ----------------------------

# GLOBAL COMPARISON PLOT
plt.figure(figsize=(6,5))

plt.scatter(np.log(df["lambda2"]), np.log(df["mean_fixation"]), label="λ2", alpha=0.7)
plt.scatter(np.log(df["kirchhoff"]), np.log(df["mean_fixation"]), label="Kirchhoff", alpha=0.7)

plt.xlabel("log(Structural Metric)")
plt.ylabel("log(Fixation Time)")
plt.title("Global Comparison: Spectral Gap vs Resistance")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("figures/global_comparison.pdf", dpi=300)
plt.savefig("figures/global_comparison.png", dpi=300)

# ----------------------------
# MULTI-PANEL (λ2)
# ----------------------------
graphs = df["graph"].unique()
n = len(graphs)

cols = 3
rows = int(np.ceil(n / cols))

fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
axes = axes.flatten()

for i, g in enumerate(graphs):
    sub = df[df["graph"] == g]

    axes[i].scatter(np.log(sub["lambda2"]), np.log(sub["mean_fixation"]))
    axes[i].set_title(g)
    axes[i].set_xlabel("log(λ2)")
    axes[i].set_ylabel("log(T)")

# remove empty axes
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("figures/lambda_panels.pdf", dpi=300)
plt.savefig("figures/lambda_panels.png", dpi=300)

# ----------------------------
# MULTI-PANEL (Kirchhoff)
# ----------------------------
fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
axes = axes.flatten()

for i, g in enumerate(graphs):
    sub = df[df["graph"] == g]

    axes[i].scatter(np.log(sub["kirchhoff"]), np.log(sub["mean_fixation"]))
    axes[i].set_title(g)
    axes[i].set_xlabel("log(Kirchhoff)")
    axes[i].set_ylabel("log(T)")

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("figures/kirchhoff_panels.pdf", dpi=300)
plt.savefig("figures/kirchhoff_panels.png", dpi=300)

print("Figures saved in /figures/")