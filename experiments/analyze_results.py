import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

# -------------------------------
# CONFIG
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_FILE = os.path.join(BASE_DIR, "data", "results_with_resistance.csv")   # make sure this exists
OUTPUT_DIR = "figures"

sns.set_theme(style="whitegrid")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(RESULT_FILE)

# Basic cleaning
df = df[(df["lambda2"] > 0) & (df["mean_fixation"] > 0)]

# Log transforms
df["log_lambda2"] = np.log(df["lambda2"])
df["log_kf"] = np.log(df["kirchhoff"])
df["log_T"] = np.log(df["mean_fixation"])

# -------------------------------
# CREATE OUTPUT DIR
# -------------------------------
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 1. GLOBAL PLOTS
# -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# ---- λ2 plot ----
ax = axes[0]
x = df["log_lambda2"]
y = df["log_T"]

slope, intercept, r_value, _, _ = linregress(x, y)

sns.scatterplot(data=df, x="log_lambda2", y="log_T",
                hue="graph", style="graph", ax=ax, s=60, legend=False)

ax.plot(x, intercept + slope * x, linestyle='--', color='black')

ax.set_title("(a) Spectral Gap", fontsize=12)
ax.set_xlabel(r"$\log(\lambda_2)$")
ax.set_ylabel(r"$\log(T_{fix})$")

ax.text(
    0.05, 0.95,
    f"$R^2 = {r_value**2:.2f}$\n"
    f"$T \\sim \\lambda_2^{{{slope:.2f}}}$",
    transform=ax.transAxes,
    verticalalignment='top'
)

# ---- Kirchhoff plot ----
ax = axes[1]
x = df["log_kf"]
y = df["log_T"]

slope, intercept, r_value, _, _ = linregress(x, y)

sns.scatterplot(data=df, x="log_kf", y="log_T",
                hue="graph", style="graph", ax=ax, s=60)

ax.plot(x, intercept + slope * x, linestyle='--', color='black')

ax.set_title("(b) Effective Resistance", fontsize=12)
ax.set_xlabel(r"$\log(K_f)$")
ax.set_ylabel(r"$\log(T_{fix})$")

ax.text(
    0.05, 0.95,
    f"$R^2 = {r_value**2:.2f}$\n"
    f"$T \\sim K_f^{{{slope:.2f}}}$",
    transform=ax.transAxes,
    verticalalignment='top'
)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/global_analysis.png", bbox_inches='tight')
plt.close()

# -------------------------------
# 2. PER-GRAPH PANELS
# -------------------------------
graphs = df["graph"].unique()

fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300)
axes = axes.flatten()

for i, g in enumerate(graphs):
    ax = axes[i]
    sub = df[df["graph"] == g]

    # Skip degenerate cases
    if sub["kirchhoff"].nunique() < 2:
        ax.set_visible(False)
        continue

    x = np.log(sub["kirchhoff"])
    y = np.log(sub["mean_fixation"])

    slope, intercept, r_value, _, _ = linregress(x, y)

    ax.scatter(x, y, s=50, edgecolor='black')
    ax.plot(x, intercept + slope * x, linestyle='--', color='black')

    ax.set_title(f"{g}", fontsize=11)

    ax.text(
        0.05, 0.9,
        f"$R^2={r_value**2:.2f}$\n"
        f"$\\alpha={slope:.2f}$",
        transform=ax.transAxes,
        fontsize=9
    )

    ax.set_xlabel(r"$\log(K_f)$")
    ax.set_ylabel(r"$\log(T)$")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/per_graph_analysis.png", bbox_inches='tight')
plt.close()

# -------------------------------
# 3. ⭐ COMBINED MODEL (MAIN RESULT)
# -------------------------------

# Model: log(T) = a*log(Kf) + b*log(lambda2) + c
X = df[["log_kf", "log_lambda2"]]
y = df["log_T"]

model = LinearRegression()
model.fit(X, y)

r2 = model.score(X, y)
coef_kf, coef_lambda = model.coef_
intercept = model.intercept_

# Predictions
y_pred = model.predict(X)

# Plot predicted vs actual
plt.figure(figsize=(6, 6), dpi=300)

plt.scatter(y, y_pred, s=60, edgecolor='black')

# Perfect fit line
min_val = min(y.min(), y_pred.min())
max_val = max(y.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')

plt.xlabel(r"Actual $\log(T)$")
plt.ylabel(r"Predicted $\log(T)$")
plt.title("Combined Spectral Model")

plt.text(
    0.05, 0.95,
    f"$R^2 = {r2:.2f}$\n"
    f"$T \\sim K_f^{{{coef_kf:.2f}}} \\cdot \\lambda_2^{{{coef_lambda:.2f}}}$",
    transform=plt.gca().transAxes,
    verticalalignment='top'
)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/combined_model.png", bbox_inches='tight')
plt.close()

# -------------------------------
# PRINT MODEL SUMMARY
# -------------------------------
print("\n===== COMBINED MODEL =====")
print(f"R^2: {r2:.4f}")
print(f"Coefficient (log Kf): {coef_kf:.4f}")
print(f"Coefficient (log lambda2): {coef_lambda:.4f}")
print(f"Intercept: {intercept:.4f}")