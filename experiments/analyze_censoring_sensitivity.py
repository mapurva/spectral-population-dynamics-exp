"""
Censoring sensitivity analysis for Spectral Population Dynamics.

Compares two configuration-level summaries of fixation time:

1. Successful-only mean:
       mean_successful = mean(T | T <= max_steps)
   This is the statistic used in the current analysis.

2. Capped mean:
       mean_capped = [sum(successful T) + censored * max_steps] / trials
   Censored runs are assigned the simulation cap. This is a sensitivity
   summary, NOT an unbiased estimator of the uncensored mean fixation time.

For each response summary, the script evaluates:
- single-predictor log-linear models
- combined log-linear model
- 5-fold shuffled cross-validation
- leave-one-family-out cross-validation
- differences in fitted predictions

The goal is to determine whether the empirical conclusions are sensitive
to the treatment of right-censored simulations.

Input:
    data/results_revision.csv

Output directory:
    data/revision_diagnostics/censoring_sensitivity/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression


INPUT = Path("data/results_revision.csv")
OUTDIR = Path("data/revision_diagnostics/censoring_sensitivity")
OUTDIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 20260928
N_SPLITS = 5

PREDICTORS = ["lambda2", "kirchhoff"]

REQUIRED = {
    "graph",
    "n",
    "lambda2",
    "kirchhoff",
    "mean_fixation",
    "success",
    "censored",
    "trials",
    "max_steps",
}


def validate_input(df):
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not np.all(df["success"] + df["censored"] == df["trials"]):
        raise ValueError("success + censored != trials.")

    if (df["censored"] > 0).any():
        # max_steps should be defined for every row.
        if df["max_steps"].isna().any():
            raise ValueError("max_steps contains missing values.")


def add_capped_mean(df):
    """
    Reconstruct the sum of successful fixation times from:
        successful_mean * successful_count

    Then assign max_steps to every censored run.

    This is exact for the capped summary given the available aggregate
    results, assuming mean_fixation is the arithmetic mean of successful
    fixation times.
    """
    df = df.copy()

    successful_sum = df["mean_fixation"] * df["success"]

    df["mean_capped"] = (
        successful_sum
        + df["censored"] * df["max_steps"]
    ) / df["trials"]

    return df


def prepare_response(df, response):
    values = pd.to_numeric(df[response], errors="coerce").to_numpy()
    valid = np.isfinite(values) & (values > 0)

    if not np.all(valid):
        raise ValueError(f"Invalid values in response {response}.")

    return np.log(values)


def fit_ols(y, X):
    X_sm = sm.add_constant(X)
    return sm.OLS(y, X_sm).fit()


def in_sample_models(df, response):
    y = prepare_response(df, response)

    rows = []

    for predictor in PREDICTORS:
        x = np.log(df[predictor].to_numpy())

        model = fit_ols(y, x)

        pred = model.predict(sm.add_constant(x))
        resid = y - pred

        rows.append(
            {
                "response": response,
                "model": predictor,
                "n": len(y),
                "slope": model.params[1],
                "intercept": model.params[0],
                "r2": model.rsquared,
                "rmse_log": np.sqrt(np.mean(resid ** 2)),
                "mae_log": np.mean(np.abs(resid)),
            }
        )

    X = np.column_stack(
        [
            np.log(df["kirchhoff"].to_numpy()),
            np.log(df["lambda2"].to_numpy()),
        ]
    )

    model = fit_ols(y, X)
    pred = model.predict(sm.add_constant(X))
    resid = y - pred

    rows.append(
        {
            "response": response,
            "model": "combined",
            "n": len(y),
            "slope_kirchhoff": model.params[1],
            "slope_lambda2": model.params[2],
            "intercept": model.params[0],
            "r2": model.rsquared,
            "rmse_log": np.sqrt(np.mean(resid ** 2)),
            "mae_log": np.mean(np.abs(resid)),
        }
    )

    return pd.DataFrame(rows)


def cv_metrics(y, X, model_name, response):
    """
    Ordinary shuffled 5-fold CV over the 30 configurations.
    """
    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    r2_values = []
    rmse_values = []
    mae_values = []

    for train_idx, test_idx in kfold.split(X):
        model = LinearRegression()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])

        r2_values.append(r2_score(y[test_idx], pred))
        rmse_values.append(np.sqrt(mean_squared_error(y[test_idx], pred)))
        mae_values.append(mean_absolute_error(y[test_idx], pred))

    return {
        "response": response,
        "model": model_name,
        "cv_r2_mean": np.mean(r2_values),
        "cv_r2_std": np.std(r2_values),
        "cv_rmse_mean": np.mean(rmse_values),
        "cv_rmse_std": np.std(rmse_values),
        "cv_mae_mean": np.mean(mae_values),
        "cv_mae_std": np.std(mae_values),
    }


def cross_validation(df, response):
    y = prepare_response(df, response)

    X_k = np.log(df[["kirchhoff"]].to_numpy())
    X_l = np.log(df[["lambda2"]].to_numpy())
    X_c = np.log(df[["kirchhoff", "lambda2"]].to_numpy())

    rows = [
        cv_metrics(y, X_k, "kirchhoff", response),
        cv_metrics(y, X_l, "lambda2", response),
        cv_metrics(y, X_c, "combined", response),
    ]

    return pd.DataFrame(rows)


def leave_one_family_out(df, response):
    y_all = prepare_response(df, response)

    rows = []

    for held_out in sorted(df["graph"].unique()):
        train = df["graph"] != held_out
        test = ~train

        X_train = np.log(
            df.loc[train, ["kirchhoff", "lambda2"]].to_numpy()
        )
        y_train = y_all[train.to_numpy()]

        X_test = np.log(
            df.loc[test, ["kirchhoff", "lambda2"]].to_numpy()
        )
        y_test = y_all[test.to_numpy()]

        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        rows.append(
            {
                "response": response,
                "held_out_family": held_out,
                "n_test": len(y_test),
                "r2": r2_score(y_test, pred),
                "rmse_log": np.sqrt(mean_squared_error(y_test, pred)),
                "mae_log": mean_absolute_error(y_test, pred),
            }
        )

    return pd.DataFrame(rows)


def prediction_difference(df):
    """
    Compare predictions from the combined model using the two response
    definitions, back on the original fixation-time scale.
    """
    rows = []

    X = np.log(df[["kirchhoff", "lambda2"]].to_numpy())

    y_success = np.log(df["mean_fixation"].to_numpy())
    y_capped = np.log(df["mean_capped"].to_numpy())

    model_success = LinearRegression().fit(X, y_success)
    model_capped = LinearRegression().fit(X, y_capped)

    pred_success = np.exp(model_success.predict(X))
    pred_capped = np.exp(model_capped.predict(X))

    for i, row in df.iterrows():
        rows.append(
            {
                "graph": row["graph"],
                "n": row["n"],
                "censor_rate": row["censored"] / row["trials"],
                "mean_successful": row["mean_fixation"],
                "mean_capped": row["mean_capped"],
                "pred_success_model": pred_success[i],
                "pred_capped_model": pred_capped[i],
                "relative_change_response": (
                    row["mean_capped"] / row["mean_fixation"] - 1.0
                ),
                "relative_change_prediction": (
                    pred_capped[i] / pred_success[i] - 1.0
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT}. Run the revised experiment first."
        )

    df = pd.read_csv(INPUT)
    validate_input(df)
    df = add_capped_mean(df)

    # Save configuration-level comparison.
    df[
        [
            "graph",
            "n",
            "success",
            "censored",
            "trials",
            "max_steps",
            "mean_fixation",
            "mean_capped",
        ]
    ].to_csv(OUTDIR / "configuration_censoring_comparison.csv", index=False)

    all_in_sample = pd.concat(
        [
            in_sample_models(df, "mean_fixation"),
            in_sample_models(df, "mean_capped"),
        ],
        ignore_index=True,
    )
    all_in_sample.to_csv(
        OUTDIR / "in_sample_models.csv",
        index=False,
    )

    all_cv = pd.concat(
        [
            cross_validation(df, "mean_fixation"),
            cross_validation(df, "mean_capped"),
        ],
        ignore_index=True,
    )
    all_cv.to_csv(OUTDIR / "five_fold_cv.csv", index=False)

    all_lofo = pd.concat(
        [
            leave_one_family_out(df, "mean_fixation"),
            leave_one_family_out(df, "mean_capped"),
        ],
        ignore_index=True,
    )
    all_lofo.to_csv(
        OUTDIR / "leave_one_family_out.csv",
        index=False,
    )

    pred_diff = prediction_difference(df)
    pred_diff.to_csv(
        OUTDIR / "prediction_sensitivity.csv",
        index=False,
    )

    # Compact console report.
    print("=" * 80)
    print("CONFIGURATION-LEVEL CENSORING SENSITIVITY")
    print("=" * 80)
    print(
        df[
            [
                "graph",
                "n",
                "censored",
                "trials",
                "mean_fixation",
                "mean_capped",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )

    print("\n" + "=" * 80)
    print("IN-SAMPLE MODELS")
    print("=" * 80)
    print(
        all_in_sample.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )

    print("\n" + "=" * 80)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 80)
    print(
        all_cv.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )

    print("\n" + "=" * 80)
    print("LEAVE-ONE-FAMILY-OUT: COMBINED MODEL")
    print("=" * 80)
    print(
        all_lofo[all_lofo["model"] if "model" in all_lofo else all_lofo.columns[0]:]
        if False else
        all_lofo.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )

    print("\n" + "=" * 80)
    print("LARGEST RESPONSE CHANGES")
    print("=" * 80)
    largest = pred_diff.sort_values(
        "relative_change_response",
        ascending=False,
    ).head(10)

    print(
        largest[
            [
                "graph",
                "n",
                "censor_rate",
                "mean_successful",
                "mean_capped",
                "relative_change_response",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )

    print("\nSaved results to:")
    print(OUTDIR.resolve())


if __name__ == "__main__":
    main()
