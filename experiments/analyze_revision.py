import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


INPUT = "data/results_revision.csv"


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT)

# The response variable is the observed mean fixation time
# among successful (non-censored) trials.
df = df.copy()

df["log_T"] = np.log(df["mean_fixation"])
df["log_K"] = np.log(df["kirchhoff"])
df["log_lambda2"] = np.log(df["lambda2"])

df["censor_rate"] = (
    df["censored"] / df["trials"]
)


print("=" * 80)
print("REVISED EXPERIMENT: STATISTICAL ANALYSIS")
print("=" * 80)


# ============================================================
# 1. Censoring summary
# ============================================================

print("\n" + "=" * 80)
print("1. CENSORING")
print("=" * 80)

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

print("\nFamily-level censoring:")

family_censor = (
    df.groupby("graph")
    .agg(
        trials=("trials", "sum"),
        censored=("censored", "sum"),
    )
)

family_censor["censor_rate"] = (
    family_censor["censored"]
    / family_censor["trials"]
)

print(family_censor)


# ============================================================
# 2. Descriptive statistics
# ============================================================

print("\n" + "=" * 80)
print("2. DESCRIPTIVE STATISTICS")
print("=" * 80)

print(
    df[
        [
            "graph",
            "n",
            "lambda2",
            "kirchhoff",
            "mean_fixation",
            "std",
            "diameter",
            "avg_path",
            "degree_var",
        ]
    ].to_string(index=False)
)


# ============================================================
# 3. Correlations among candidate predictors
# ============================================================

print("\n" + "=" * 80)
print("3. PREDICTOR CORRELATIONS")
print("=" * 80)

predictors = [
    "log_lambda2",
    "log_K",
    "diameter",
    "avg_path",
    "degree_var",
]

print(
    df[predictors].corr().round(4)
)


# ============================================================
# 4. Simple log-log models
# ============================================================

def fit_ols(x, y):
    model = LinearRegression()
    model.fit(x, y)

    pred = model.predict(x)

    return {
        "model": model,
        "r2": r2_score(y, pred),
        "rmse": np.sqrt(mean_squared_error(y, pred)),
        "mae": mean_absolute_error(y, pred),
        "coef": model.coef_,
        "intercept": model.intercept_,
    }


X_lambda = df[["log_lambda2"]].values
X_K = df[["log_K"]].values
X_combined = df[["log_K", "log_lambda2"]].values
y = df["log_T"].values


lambda_model = fit_ols(X_lambda, y)
K_model = fit_ols(X_K, y)
combined_model = fit_ols(X_combined, y)


print("\n" + "=" * 80)
print("4. IN-SAMPLE LOG-LOG MODELS")
print("=" * 80)

print(
    f"Spectral gap:\n"
    f"  R2   = {lambda_model['r2']:.6f}\n"
    f"  RMSE = {lambda_model['rmse']:.6f}\n"
    f"  MAE  = {lambda_model['mae']:.6f}\n"
    f"  coef = {lambda_model['coef'][0]:.6f}"
)

print(
    f"\nKirchhoff index:\n"
    f"  R2   = {K_model['r2']:.6f}\n"
    f"  RMSE = {K_model['rmse']:.6f}\n"
    f"  MAE  = {K_model['mae']:.6f}\n"
    f"  coef = {K_model['coef'][0]:.6f}"
)

print(
    f"\nCombined model:\n"
    f"  R2   = {combined_model['r2']:.6f}\n"
    f"  RMSE = {combined_model['rmse']:.6f}\n"
    f"  MAE  = {combined_model['mae']:.6f}\n"
    f"  coef log_K       = {combined_model['coef'][0]:.6f}\n"
    f"  coef log_lambda2 = {combined_model['coef'][1]:.6f}"
)


# ============================================================
# 5. VIF
# ============================================================

print("\n" + "=" * 80)
print("5. MULTICOLLINEARITY / VIF")
print("=" * 80)

X_vif = sm.add_constant(
    df[["log_K", "log_lambda2"]]
)

vif = pd.DataFrame({
    "variable": X_vif.columns,
    "VIF": [
        variance_inflation_factor(
            X_vif.values,
            i
        )
        for i in range(X_vif.shape[1])
    ],
})

print(vif)


# ============================================================
# 6. 5-fold cross-validation
# ============================================================

print("\n" + "=" * 80)
print("6. FIVE-FOLD CROSS-VALIDATION")
print("=" * 80)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=20260928
)


def cross_validate_model(X, y, name):
    model = LinearRegression()

    r2_scores = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="r2"
    )

    rmse_scores = np.sqrt(
        -cross_val_score(
            model,
            X,
            y,
            cv=kf,
            scoring="neg_mean_squared_error"
        )
    )

    mae_scores = -cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="neg_mean_absolute_error"
    )

    print(f"\n{name}")
    print(
        f"  CV R2:   "
        f"{r2_scores.mean():.6f} "
        f"+/- {r2_scores.std():.6f}"
    )
    print(
        f"  CV RMSE: "
        f"{rmse_scores.mean():.6f} "
        f"+/- {rmse_scores.std():.6f}"
    )
    print(
        f"  CV MAE:  "
        f"{mae_scores.mean():.6f} "
        f"+/- {mae_scores.std():.6f}"
    )


cross_validate_model(
    X_lambda,
    y,
    "Spectral gap"
)

cross_validate_model(
    X_K,
    y,
    "Kirchhoff index"
)

cross_validate_model(
    X_combined,
    y,
    "Combined model"
)


# ============================================================
# 7. Leave-one-family-out validation
# ============================================================

print("\n" + "=" * 80)
print("7. LEAVE-ONE-GRAPH-FAMILY-OUT VALIDATION")
print("=" * 80)

families = sorted(df["graph"].unique())

for held_out in families:

    train = df[df["graph"] != held_out]
    test = df[df["graph"] == held_out]

    model = LinearRegression()

    model.fit(
        train[["log_K", "log_lambda2"]],
        train["log_T"]
    )

    prediction = model.predict(
        test[["log_K", "log_lambda2"]]
    )

    r2 = r2_score(
        test["log_T"],
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            test["log_T"],
            prediction
        )
    )

    mae = mean_absolute_error(
        test["log_T"],
        prediction
    )

    print(
        f"{held_out:12s} "
        f"R2={r2: .6f} "
        f"RMSE={rmse: .6f} "
        f"MAE={mae: .6f}"
    )


# ============================================================
# 8. Censoring sensitivity
# ============================================================

print("\n" + "=" * 80)
print("8. CENSORING SENSITIVITY")
print("=" * 80)

for threshold in [0.00, 0.01, 0.05, 0.10, 0.25]:

    subset = df[
        df["censor_rate"] <= threshold
    ]

    if len(subset) < 5:
        continue

    model = LinearRegression()

    X = subset[
        ["log_K", "log_lambda2"]
    ]

    model.fit(
        X,
        subset["log_T"]
    )

    prediction = model.predict(X)

    r2 = r2_score(
        subset["log_T"],
        prediction
    )

    print(
        f"Configurations with censoring <= "
        f"{threshold * 100:.0f}%: "
        f"{len(subset):2d} "
        f"R2={r2:.6f}"
    )


print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)