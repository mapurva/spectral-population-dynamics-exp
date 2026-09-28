import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression

# -------------------------------
# CONFIG (ROBUST PATHS)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "results_with_resistance.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(DATA_FILE)

df = df[(df["lambda2"] > 0) & (df["mean_fixation"] > 0) & (df["kirchhoff"] > 0)]

# Log transforms
df["log_lambda2"] = np.log(df["lambda2"])
df["log_kf"] = np.log(df["kirchhoff"])
df["log_T"] = np.log(df["mean_fixation"])

print(f"Loaded {len(df)} valid samples")

# -------------------------------
# GLOBAL AXIS LIMITS
# -------------------------------
lambda_xlim = (df["log_lambda2"].min(), df["log_lambda2"].max())
kf_xlim = (df["log_kf"].min(), df["log_kf"].max())
ylim = (df["log_T"].min(), df["log_T"].max())

# -------------------------------
# 1. GLOBAL ANALYSIS
# -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# ---- Spectral Gap ----
ax = axes[0]
x = df["log_lambda2"]
y = df["log_T"]

slope, intercept, r_value, _, _ = linregress(x, y)

sns.scatterplot(
    data=df,
    x="log_lambda2",
    y="log_T",
    hue="graph",
    style="graph",
    s=60,
    legend=False,
    ax=ax
)

# sorted regression line
idx = np.argsort(x)
ax.plot(x.iloc[idx], (intercept + slope * x).iloc[idx],
        linestyle='--', color='black')

ax.set_title("(a) Spectral Gap")
ax.set_xlabel(r"$\log(\lambda_2)$")
ax.set_ylabel(r"$\log(T_{fix})$")
ax.set_xlim(lambda_xlim)
ax.set_ylim(ylim)

ax.text(
    0.05, 0.95,
    f"$R^2 = {r_value**2:.2f}$\n"
    f"slope = {slope:.2f}",
    transform=ax.transAxes,
    verticalalignment='top'
)

# ---- Kirchhoff ----
ax = axes[1]
x = df["log_kf"]
y = df["log_T"]

slope, intercept, r_value, _, _ = linregress(x, y)

sns.scatterplot(
    data=df,
    x="log_kf",
    y="log_T",
    hue="graph",
    style="graph",
    s=60,
    ax=ax
)

idx = np.argsort(x)
ax.plot(x.iloc[idx], (intercept + slope * x).iloc[idx],
        linestyle='--', color='black')

ax.set_title("(b) Effective Resistance")
ax.set_xlabel(r"$\log(K_f)$")
ax.set_ylabel(r"$\log(T_{fix})$")
ax.set_xlim(kf_xlim)
ax.set_ylim(ylim)

ax.legend(loc="lower right", fontsize=8)

ax.text(
    0.05, 0.95,
    f"$R^2 = {r_value**2:.2f}$\n"
    f"slope = {slope:.2f}",
    transform=ax.transAxes,
    verticalalignment='top'
)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "global_analysis.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, "global_analysis.png"), bbox_inches='tight')
plt.close()

# -------------------------------
# 2. PER-GRAPH ANALYSIS
# -------------------------------
graphs = df["graph"].unique()

fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300)
axes = axes.flatten()

for i, g in enumerate(graphs):
    ax = axes[i]
    sub = df[df["graph"] == g]

    if sub["kirchhoff"].nunique() < 2:
        ax.set_visible(False)
        continue

    x = sub["log_kf"]
    y = sub["log_T"]

    slope, intercept, r_value, _, _ = linregress(x, y)

    ax.scatter(x, y, s=50, edgecolor='black')

    idx = np.argsort(x)
    ax.plot(x.iloc[idx], (intercept + slope * x).iloc[idx],
            linestyle='--', color='black')

    ax.set_title(f"{g} (n=20–120)")
    ax.set_xlabel(r"$\log(K_f)$")
    ax.set_ylabel(r"$\log(T_{fix})$")
    ax.set_xlim(kf_xlim)
    ax.set_ylim(ylim)

    ax.text(
        0.05, 0.9,
        f"$R^2={r_value**2:.2f}$\n"
        f"slope={slope:.2f}",
        transform=ax.transAxes,
        fontsize=9
    )

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "per_graph_analysis.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, "per_graph_analysis.png"), bbox_inches='tight')
plt.close()

# -------------------------------
# 3. COMBINED MODEL (MAIN RESULT)
# -------------------------------
X = df[["log_kf", "log_lambda2"]]
y = df["log_T"]

model = LinearRegression()
model.fit(X, y)

r2 = model.score(X, y)
coef_kf, coef_lambda = model.coef_

y_pred = model.predict(X)

plt.figure(figsize=(6, 6), dpi=300)

sns.scatterplot(
    x=y,
    y=y_pred,
    hue=df["graph"],
    s=60,
    edgecolor='black'
)

min_val = min(y.min(), y_pred.min())
max_val = max(y.max(), y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black')

plt.xlabel(r"Actual $\log(T_{fix})$")
plt.ylabel(r"Predicted $\log(T_{fix})$")
plt.title("Combined Spectral-Resistance Model")

plt.text(
    0.05, 0.95,
    f"$R^2 = {r2:.2f}$\n"
    f"$\\log T = {coef_kf:.2f} \\log K_f + {coef_lambda:.2f} \\log \\lambda_2$",
    transform=plt.gca().transAxes,
    verticalalignment='top'
)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "combined_model.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, "combined_model.png"), bbox_inches='tight')
plt.close()

# -------------------------------
# PRINT SUMMARY
# -------------------------------
print("\n===== COMBINED MODEL =====")
print(f"R^2: {r2:.4f}")
print(f"Coefficient (log Kf): {coef_kf:.4f}")
print(f"Coefficient (log lambda2): {coef_lambda:.4f}")