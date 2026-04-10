import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# ----------------------------
# SETUP
# ----------------------------
os.makedirs("data/analysis", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ----------------------------
# LOAD DATA (FIXED)
# ----------------------------
try:
    df = pd.read_csv("data/results_with_resistance.csv")
except FileNotFoundError:
    raise FileNotFoundError("Run experiments first: results_with_resistance.csv not found")

# Clean data
df = df.dropna()
df = df[
    (df["mean_fixation"] > 0) &
    (df["lambda2"] > 0) &
    (df["kirchhoff"] > 0)
]

print(f"Loaded {len(df)} valid rows")

# ----------------------------
# REGRESSION WITH CI
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
# SAVE ANALYSIS TABLE
# ----------------------------
records = []

# Global
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

# Per graph
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

results_df = pd.DataFrame(records)
results_df.to_csv("data/analysis/analysis_results.csv", index=False)

print("Saved: data/analysis/analysis_results.csv")

# ----------------------------
# PLOTTING (FIXED + IMPROVED)
# ----------------------------

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 11
})

# Helper: regression line + R²
def add_regression(ax, x, y):
    x_log = np.log(x)
    y_log = np.log(y)

    slope, intercept = np.polyfit(x_log, y_log, 1)
    r = np.corrcoef(x_log, y_log)[0, 1]
    r2 = r**2

    x_line = np.linspace(x_log.min(), x_log.max(), 100)
    y_line = slope * x_line + intercept

    ax.plot(x_line, y_line, linestyle="--")
    ax.text(0.05, 0.9, f"$R^2$={r2:.2f}", transform=ax.transAxes)


# ----------------------------
# GLOBAL SPLIT FIGURE
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# λ2
axes[0].scatter(np.log(df["lambda2"]), np.log(df["mean_fixation"]))
add_regression(axes[0], df["lambda2"], df["mean_fixation"])
axes[0].set_xlabel("log(λ₂)")
axes[0].set_ylabel("log(Fixation Time)")
axes[0].set_title("(a) Spectral Gap")

# Kirchhoff
axes[1].scatter(np.log(df["kirchhoff"]), np.log(df["mean_fixation"]))
add_regression(axes[1], df["kirchhoff"], df["mean_fixation"])
axes[1].set_xlabel("log(Kirchhoff Index)")
axes[1].set_ylabel("log(Fixation Time)")
axes[1].set_title("(b) Effective Resistance")

plt.tight_layout()
plt.savefig("figures/global_split.pdf", dpi=300)
plt.savefig("figures/global_split.png", dpi=300)
plt.close()

# ----------------------------
# MULTI-PANEL λ2
# ----------------------------
graphs = df["graph"].unique()
cols = 3
rows = int(np.ceil(len(graphs) / cols))

fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
axes = axes.flatten()

for i, g in enumerate(graphs):
    sub = df[df["graph"] == g]

    x = sub["lambda2"]
    y = sub["mean_fixation"]

    ax = axes[i]
    ax.scatter(np.log(x), np.log(y))
    add_regression(ax, x, y)

    ax.set_title(g)
    ax.set_xlabel("log(λ₂)")
    ax.set_ylabel("log(T)")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("figures/lambda_panels_refined.pdf", dpi=300)
plt.savefig("figures/lambda_panels_refined.png", dpi=300)
plt.close()

# ----------------------------
# MULTI-PANEL KIRCHHOFF
# ----------------------------
fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
axes = axes.flatten()

for i, g in enumerate(graphs):
    sub = df[df["graph"] == g]

    x = sub["kirchhoff"]
    y = sub["mean_fixation"]

    ax = axes[i]
    ax.scatter(np.log(x), np.log(y))
    add_regression(ax, x, y)

    ax.set_title(g)
    ax.set_xlabel("log(Kirchhoff)")
    ax.set_ylabel("log(T)")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("figures/kirchhoff_panels_refined.pdf", dpi=300)
plt.savefig("figures/kirchhoff_panels_refined.png", dpi=300)
plt.close()

print("All refined figures saved in /figures/")