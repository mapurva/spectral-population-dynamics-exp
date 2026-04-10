import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load data
df = pd.read_csv("data/results_with_resistance.csv")

# Clean
df = df.dropna()
df = df[(df["mean_fixation"] > 0) & (df["lambda2"] > 0) & (df["kirchhoff"] > 0)]

# ----------------------------
# Helper: regression with CI
# ----------------------------
def regression_with_ci(x, y):
    x_log = np.log(x)
    y_log = np.log(y)

    slope, intercept, r, p, stderr = linregress(x_log, y_log)

    alpha = -slope
    ci_low = alpha - 1.96 * stderr
    ci_high = alpha + 1.96 * stderr

    return {
        "alpha": alpha,
        "r2": r**2,
        "p": p,
        "ci": (ci_low, ci_high)
    }

# ----------------------------
# GLOBAL MODELS
# ----------------------------
print("\n=== GLOBAL MODELS ===")

models = {
    "lambda2": df["lambda2"],
    "kirchhoff": df["kirchhoff"]
}

for name, x in models.items():
    res = regression_with_ci(x, df["mean_fixation"])
    print(f"\nModel: {name}")
    print(res)

# ----------------------------
# PER GRAPH ANALYSIS
# ----------------------------
print("\n=== PER GRAPH ANALYSIS ===")

results = {}

for g in df["graph"].unique():
    sub = df[df["graph"] == g]

    if len(sub) < 3:
        continue

    res_lambda = regression_with_ci(sub["lambda2"], sub["mean_fixation"])
    res_kirchhoff = regression_with_ci(sub["kirchhoff"], sub["mean_fixation"])

    results[g] = {
        "lambda2": res_lambda,
        "kirchhoff": res_kirchhoff
    }

    print(f"\nGraph: {g}")
    print("λ2:", res_lambda)
    print("Kirchhoff:", res_kirchhoff)

# ----------------------------
# PLOTTING
# ----------------------------

# 1. Global scatter
plt.figure()
plt.scatter(np.log(df["lambda2"]), np.log(df["mean_fixation"]))
plt.xlabel("log(lambda2)")
plt.ylabel("log(fixation time)")
plt.title("Global: λ2 vs Fixation Time")
plt.show()

# 2. Per graph plots
for g in df["graph"].unique():
    sub = df[df["graph"] == g]

    if len(sub) < 3:
        continue

    plt.figure()
    plt.scatter(np.log(sub["lambda2"]), np.log(sub["mean_fixation"]))
    plt.xlabel("log(lambda2)")
    plt.ylabel("log(fixation time)")
    plt.title(f"{g}: λ2 vs Fixation Time")
    plt.show()

# 3. Kirchhoff plots
for g in df["graph"].unique():
    sub = df[df["graph"] == g]

    if len(sub) < 3:
        continue

    plt.figure()
    plt.scatter(np.log(sub["kirchhoff"]), np.log(sub["mean_fixation"]))
    plt.xlabel("log(Kirchhoff)")
    plt.ylabel("log(fixation time)")
    plt.title(f"{g}: Kirchhoff vs Fixation Time")
    plt.show()