from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "results_revision.csv"
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = FIGURE_DIR / "figure_2_observed_vs_fitted.png"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
df = pd.read_csv(DATA_FILE)

required_columns = [
    "graph",
    "mean_fixation",
    "lambda2",
    "kirchhoff",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns in {DATA_FILE}: {missing}"
    )


# ---------------------------------------------------------
# Prepare variables
# ---------------------------------------------------------
df = df[
    (df["mean_fixation"] > 0)
    & (df["kirchhoff"] > 0)
    & (df["lambda2"] > 0)
].copy()

df["log_T"] = np.log(df["mean_fixation"])
df["log_K"] = np.log(df["kirchhoff"])
df["log_lambda2"] = np.log(df["lambda2"])


# ---------------------------------------------------------
# Fit combined pooled model
#
# log(T) = beta0 + beta1 log(K) + beta2 log(lambda2)
# ---------------------------------------------------------
X = df[["log_K", "log_lambda2"]]
X = sm.add_constant(X)

y = df["log_T"]

model = sm.OLS(y, X).fit()

df["fitted_log_T"] = model.predict(X)

r2 = model.rsquared


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 6.2))

families = [
    "cycle",
    "complete",
    "erdos_renyi",
    "path",
    "star",
    "barbell",
]

markers = [
    "o",
    "s",
    "^",
    "D",
    "P",
    "X",
]

for family, marker in zip(families, markers):

    subset = df[df["graph"].str.lower() == family]

    if subset.empty:
        continue

    ax.scatter(
        subset["log_T"],
        subset["fitted_log_T"],
        marker=marker,
        s=55,
        label=family.replace("_", " ").title(),
        alpha=0.85,
    )


# ---------------------------------------------------------
# 45-degree reference line
# ---------------------------------------------------------
all_values = np.concatenate(
    [
        df["log_T"].values,
        df["fitted_log_T"].values,
    ]
)

line_min = all_values.min()
line_max = all_values.max()

margin = 0.15 * (line_max - line_min)

line_min -= margin
line_max += margin

ax.plot(
    [line_min, line_max],
    [line_min, line_max],
    linestyle="--",
    linewidth=1.2,
)


# ---------------------------------------------------------
# Labels and formatting
# ---------------------------------------------------------
ax.set_xlabel("Observed log(mean fixation time)")
ax.set_ylabel("Fitted log(mean fixation time)")

ax.set_title(
    "Observed versus fitted fixation time\n"
    "Combined Kirchhoff-index and spectral-gap model"
)

ax.text(
    0.05,
    0.95,
    f"R² = {r2:.3f}",
    transform=ax.transAxes,
    verticalalignment="top",
)

ax.legend(
    title="Graph family",
    fontsize=8,
    title_fontsize=9,
)

ax.grid(True, alpha=0.25)

fig.tight_layout()


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
fig.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Figure generated successfully.")
print(f"Output: {OUTPUT_FILE}")
print(f"Model R²: {r2:.6f}")
print(
    "Model coefficients: "
    f"intercept={model.params['const']:.6f}, "
    f"log(K)={model.params['log_K']:.6f}, "
    f"log(lambda2)={model.params['log_lambda2']:.6f}"
)