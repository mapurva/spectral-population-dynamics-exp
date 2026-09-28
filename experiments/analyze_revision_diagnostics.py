"""
Revision diagnostics for Spectral Population Dynamics.

Purpose
-------
1. Diagnose family-wise scaling and identifiability.
2. Compare structural metrics on the same 30 graph configurations.
3. Flag degenerate/near-degenerate predictors before interpreting slopes.
4. Summarize censoring concentration.

Input
-----
data/results_revision.csv

Expected columns
----------------
graph,n,lambda2,mean_fixation,std,success,censored,trials,max_steps,
graph_seed,simulation_seed,diameter,avg_path,degree_var,kirchhoff

This script intentionally does NOT claim a universal scaling law.
Family-wise regressions are descriptive diagnostics only.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm


INPUT = Path("data/results_revision.csv")
OUTDIR = Path("data/revision_diagnostics")
OUTDIR.mkdir(parents=True, exist_ok=True)

METRICS = ["lambda2", "kirchhoff", "diameter", "avg_path", "degree_var"]


def fit_log_model(df, predictor):
    """Fit log(mean_fixation) ~ log(predictor), if identifiable."""
    x = pd.to_numeric(df[predictor], errors="coerce")
    y = pd.to_numeric(df["mean_fixation"], errors="coerce")

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > 0)
        & (y > 0)
    )

    x = np.log(x[valid].to_numpy())
    y = np.log(y[valid].to_numpy())

    unique_x = np.unique(x)

    if len(x) < 3:
        return {
            "n": len(x),
            "unique_x": len(unique_x),
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "status": "insufficient observations",
        }

    if len(unique_x) < 2 or np.ptp(x) < 1e-12:
        return {
            "n": len(x),
            "unique_x": len(unique_x),
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "status": "non-identifiable: predictor has zero variance",
        }

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    return {
        "n": len(x),
        "unique_x": len(unique_x),
        "slope": model.params[1],
        "intercept": model.params[0],
        "r2": model.rsquared,
        "status": "identified",
    }


def family_diagnostics(df):
    rows = []

    for family, gdf in df.groupby("graph", sort=True):
        for predictor in ["lambda2", "kirchhoff"]:
            result = fit_log_model(gdf, predictor)
            result.update(
                {
                    "graph": family,
                    "predictor": predictor,
                    "n_configs": len(gdf),
                    "predictor_min": gdf[predictor].min(),
                    "predictor_max": gdf[predictor].max(),
                    "predictor_cv": (
                        gdf[predictor].std(ddof=1) / gdf[predictor].mean()
                        if gdf[predictor].mean() != 0
                        else np.nan
                    ),
                }
            )
            rows.append(result)

    out = pd.DataFrame(rows)
    return out[
        [
            "graph",
            "predictor",
            "n_configs",
            "n",
            "unique_x",
            "predictor_min",
            "predictor_max",
            "predictor_cv",
            "slope",
            "intercept",
            "r2",
            "status",
        ]
    ]


def baseline_comparison(df):
    rows = []

    y = np.log(df["mean_fixation"].to_numpy())

    for predictor in METRICS:
        x = pd.to_numeric(df[predictor], errors="coerce").to_numpy()

        valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
        xv = np.log(x[valid])
        yv = y[valid]

        if len(np.unique(xv)) < 2:
            rows.append(
                {
                    "predictor": predictor,
                    "n": len(xv),
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r2": np.nan,
                    "rmse_log": np.nan,
                    "mae_log": np.nan,
                    "status": "non-identifiable",
                }
            )
            continue

        X = sm.add_constant(xv)
        model = sm.OLS(yv, X).fit()
        pred = model.predict(X)
        resid = yv - pred

        rows.append(
            {
                "predictor": predictor,
                "n": len(xv),
                "slope": model.params[1],
                "intercept": model.params[0],
                "r2": model.rsquared,
                "rmse_log": np.sqrt(np.mean(resid ** 2)),
                "mae_log": np.mean(np.abs(resid)),
                "status": "identified",
            }
        )

    # Pairwise correlation among candidate structural metrics.
    log_df = pd.DataFrame(index=df.index)
    for predictor in METRICS:
        values = pd.to_numeric(df[predictor], errors="coerce")
        log_df[f"log_{predictor}"] = np.log(values.where(values > 0))

    corr = log_df.corr()

    return pd.DataFrame(rows), corr


def censoring_summary(df):
    tmp = df.copy()
    tmp["censor_rate"] = tmp["censored"] / tmp["trials"]

    by_family = (
        tmp.groupby("graph")
        .agg(
            configs=("n", "count"),
            trials=("trials", "sum"),
            censored=("censored", "sum"),
            success=("success", "sum"),
        )
        .reset_index()
    )
    by_family["censor_rate"] = (
        by_family["censored"] / by_family["trials"]
    )

    by_config = tmp[
        [
            "graph",
            "n",
            "success",
            "censored",
            "trials",
            "censor_rate",
            "mean_fixation",
        ]
    ].copy()

    return by_family, by_config


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT}. Run the revised experiment first."
        )

    df = pd.read_csv(INPUT)

    required = {
        "graph",
        "n",
        "lambda2",
        "mean_fixation",
        "success",
        "censored",
        "trials",
        "diameter",
        "avg_path",
        "degree_var",
        "kirchhoff",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Basic integrity checks.
    if not np.all(df["success"] + df["censored"] == df["trials"]):
        raise ValueError("success + censored != trials for at least one row.")

    # 1. Family-wise diagnostics.
    fam = family_diagnostics(df)
    fam.to_csv(OUTDIR / "family_scaling_diagnostics.csv", index=False)

    # 2. Baseline metric comparison.
    baseline, corr = baseline_comparison(df)
    baseline.to_csv(OUTDIR / "baseline_metric_comparison.csv", index=False)
    corr.to_csv(OUTDIR / "structural_metric_correlations.csv")

    # 3. Censoring.
    by_family, by_config = censoring_summary(df)
    by_family.to_csv(OUTDIR / "censoring_by_family.csv", index=False)
    by_config.to_csv(OUTDIR / "censoring_by_configuration.csv", index=False)

    # 4. Compact console report.
    print("=" * 78)
    print("FAMILY-WISE SCALING DIAGNOSTICS")
    print("=" * 78)
    print(
        fam.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )

    print("\n" + "=" * 78)
    print("BASELINE METRIC COMPARISON: log(mean_fixation) ~ log(metric)")
    print("=" * 78)
    print(
        baseline.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )

    print("\n" + "=" * 78)
    print("LOG-STRUCTURAL-METRIC CORRELATIONS")
    print("=" * 78)
    print(corr.round(4).to_string())

    print("\n" + "=" * 78)
    print("CENSORING BY FAMILY")
    print("=" * 78)
    print(
        by_family.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )

    print("\n" + "=" * 78)
    print("HIGH-CENSORING CONFIGURATIONS (> 5%)")
    print("=" * 78)
    high = by_config[by_config["censor_rate"] > 0.05]
    if high.empty:
        print("None.")
    else:
        print(
            high.to_string(
                index=False,
                float_format=lambda x: f"{x:.6g}",
            )
        )

    print("\nSaved diagnostics to:")
    print(OUTDIR.resolve())


if __name__ == "__main__":
    main()
