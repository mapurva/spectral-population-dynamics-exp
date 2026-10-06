"""
Generate Figure 3 for the revision manuscript.

Figure 3:
Family-specific scaling between Kirchhoff index and mean fixation time.

Each point represents one graph configuration.
Both axes are logarithmic.
A separate linear regression line is fitted for each graph family.

Input:
    data/results_revision.csv

Output:
    figures/figure_3_kirchhoff_family_scaling.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "results_revision.csv"
OUTPUT_DIR = ROOT / "figures"
OUTPUT_FILE = OUTPUT_DIR / "figure_3_kirchhoff_family_scaling.png"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "graph",
    "n",
    "kirchhoff",
    "mean_fixation",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns in {INPUT_FILE}: {sorted(missing)}"
    )


# ---------------------------------------------------------
# Prepare data
# ---------------------------------------------------------

df = df[
    (df["mean_fixation"] > 0)
    & (df["kirchhoff"] > 0)
].copy()

df["log_K"] = np.log(df["kirchhoff"])
df["log_T"] = np.log(df["mean_fixation"])


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6.5))

families = [
    "cycle",
    "complete",
    "erdos_renyi",
    "path",
    "star",
    "barbell",
]

markers = {
    "cycle": "o",
    "complete": "s",
    "erdos_renyi": "^",
    "path": "D",
    "star": "P",
    "barbell": "X",
}


for family in families:

    subset = df[df["graph"] == family].copy()

    if subset.empty:
        continue

    x = subset["log_K"].to_numpy()
    y = subset["log_T"].to_numpy()

    # Scatter points
    ax.scatter(
        x,
        y,
        marker=markers[family],
        s=65,
        alpha=0.85,
        label=family.replace("_", " ").title(),
    )

    # Family-specific regression line
    if len(subset) >= 2 and np.ptp(x) > 0:

        slope, intercept = np.polyfit(x, y, 1)

        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = intercept + slope * x_line

        ax.plot(
            x_line,
            y_line,
            linewidth=1.5,
        )


# ---------------------------------------------------------
# Labels and formatting
# ---------------------------------------------------------

ax.set_xlabel(
    "log(K(G))",
    fontsize=11,
)

ax.set_ylabel(
    "log(mean fixation time)",
    fontsize=11,
)

ax.set_title(
    "Family-specific scaling between Kirchhoff index and fixation time",
    fontsize=12,
)

ax.legend(
    title="Graph family",
    fontsize=9,
    title_fontsize=9,
)

ax.grid(
    True,
    alpha=0.25,
)

fig.tight_layout()


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fig.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print("Figure generated successfully.")
print(f"Input : {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")