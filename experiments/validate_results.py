import pandas as pd
import numpy as np


INPUT = "data/results_revision.csv"

EXPECTED_GRAPHS = [
    "cycle",
    "complete",
    "erdos_renyi",
    "path",
    "star",
    "barbell",
]

EXPECTED_SIZES = [20, 30, 50, 80, 120]

EXPECTED_COLUMNS = [
    "graph",
    "n",
    "lambda2",
    "mean_fixation",
    "std",
    "success",
    "censored",
    "trials",
    "max_steps",
    "graph_seed",
    "simulation_seed",
    "diameter",
    "avg_path",
    "degree_var",
    "kirchhoff",
]


df = pd.read_csv(INPUT)

print("=" * 70)
print("RESULT VALIDATION")
print("=" * 70)

# ------------------------------------------------------------
# Basic shape
# ------------------------------------------------------------

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

assert len(df) == 30, "Expected exactly 30 configurations."
assert list(df.columns) == EXPECTED_COLUMNS, "Unexpected columns."

print("[PASS] Dataset shape and columns")


# ------------------------------------------------------------
# Graph families and sizes
# ------------------------------------------------------------

assert sorted(df["graph"].unique()) == sorted(EXPECTED_GRAPHS)
assert sorted(df["n"].unique()) == EXPECTED_SIZES

assert (
    df.groupby("graph")["n"].nunique().eq(5).all()
), "Each graph family must have 5 population sizes."

print("[PASS] Graph families and population sizes")


# ------------------------------------------------------------
# Trial accounting
# ------------------------------------------------------------

assert df["trials"].eq(500).all()
assert df["max_steps"].eq(50000).all()

assert (
    df["success"] + df["censored"] == df["trials"]
).all(), "success + censored must equal trials."

print("[PASS] Trial accounting")


# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

required_numeric = [
    "lambda2",
    "mean_fixation",
    "std",
    "success",
    "censored",
    "trials",
    "max_steps",
    "graph_seed",
    "simulation_seed",
    "diameter",
    "avg_path",
    "degree_var",
    "kirchhoff",
]

missing = df[required_numeric].isna().sum()

print("\nMissing values:")
print(missing)

assert missing.sum() == 0, "Unexpected missing numeric values."

print("[PASS] No missing numeric values")


# ------------------------------------------------------------
# Positivity / validity
# ------------------------------------------------------------

assert (df["lambda2"] > 0).all()
assert (df["kirchhoff"] > 0).all()
assert (df["diameter"] >= 1).all()
assert (df["avg_path"] > 0).all()
assert (df["success"] >= 0).all()
assert (df["censored"] >= 0).all()

print("[PASS] Basic numerical validity")


# ------------------------------------------------------------
# Seed uniqueness
# ------------------------------------------------------------

assert df["graph_seed"].is_unique
assert df["simulation_seed"].is_unique

assert not (
    df["graph_seed"] == df["simulation_seed"]
).any()

print("[PASS] Random seeds are distinct")


# ------------------------------------------------------------
# Corrected Kirchhoff values for deterministic families
# ------------------------------------------------------------

# Complete graph K_n = n - 1
complete = df[df["graph"] == "complete"]

assert np.allclose(
    complete["kirchhoff"],
    complete["n"] - 1
)

# Path graph P_n:
# K(P_n) = (n^3 - n) / 6
path = df[df["graph"] == "path"]

expected_path = (
    path["n"] ** 3 - path["n"]
) / 6

assert np.allclose(
    path["kirchhoff"],
    expected_path
)

print("[PASS] Corrected Kirchhoff values")


# ------------------------------------------------------------
# Censoring summary
# ------------------------------------------------------------

df["censor_rate"] = (
    df["censored"] / df["trials"] * 100
)

print("\nCensoring by configuration:")
print(
    df[
        [
            "graph",
            "n",
            "success",
            "censored",
            "censor_rate",
        ]
    ].to_string(index=False)
)

print("\nCensoring by graph family:")
family_censor = (
    df.groupby("graph")
    .agg(
        total_trials=("trials", "sum"),
        censored=("censored", "sum"),
    )
)

family_censor["censor_rate_percent"] = (
    family_censor["censored"]
    / family_censor["total_trials"]
    * 100
)

print(family_censor)


# ------------------------------------------------------------
# Overall censoring
# ------------------------------------------------------------

total_trials = df["trials"].sum()
total_censored = df["censored"].sum()

print()
print(f"Total trials:   {total_trials}")
print(f"Total censored: {total_censored}")
print(
    f"Overall censoring rate: "
    f"{100 * total_censored / total_trials:.4f}%"
)

print()
print("=" * 70)
print("ALL VALIDATION CHECKS PASSED")
print("=" * 70)