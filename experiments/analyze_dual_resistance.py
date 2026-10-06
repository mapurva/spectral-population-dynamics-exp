"""
Compare ordinary spectral/resistance metrics with the degree-weighted
electrical metric induced by the invasion-process dual.

Input:
    data/results_revision.csv

The script regenerates each graph from its recorded graph_seed, computes
W_I, K_I, and W_I*K_I, and merges those quantities with the empirical
fixation-time results. It then reports in-sample log-linear fits.

This is an analysis script only: it does not rerun simulations and does
not modify the original results file.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.graphs import generate_graph
from src.resistance import invasion_metric


INPUT = Path("data/results_revision.csv")
OUTDIR = Path("data/revision_dual_metric")
OUTDIR.mkdir(parents=True, exist_ok=True)


def fit_log_model(df, predictor):
    x = pd.to_numeric(df[predictor], errors="coerce").to_numpy()
    y = pd.to_numeric(df["mean_fixation"], errors="coerce").to_numpy()

    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = np.log(x[valid])
    y = np.log(y[valid])

    if len(x) < 3 or len(np.unique(x)) < 2:
        return {
            "predictor": predictor,
            "n": len(x),
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "rmse_log": np.nan,
            "mae_log": np.nan,
            "status": "non-identifiable",
        }

    model = sm.OLS(y, sm.add_constant(x)).fit()
    pred = model.predict(sm.add_constant(x))
    resid = y - pred

    return {
        "predictor": predictor,
        "n": len(x),
        "slope": float(model.params[1]),
        "intercept": float(model.params[0]),
        "r2": float(model.rsquared),
        "rmse_log": float(np.sqrt(np.mean(resid ** 2))),
        "mae_log": float(np.mean(np.abs(resid))),
        "status": "identified",
    }


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT}. Run the revised experiment first."
        )

    df = pd.read_csv(INPUT)

    required = {
        "graph", "n", "lambda2", "mean_fixation", "graph_seed"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []

    for row in df.itertuples(index=False):
        G = generate_graph(row.graph, int(row.n), seed=int(row.graph_seed))
        metrics = invasion_metric(G)

        rows.append({
            "graph": row.graph,
            "n": int(row.n),
            "graph_seed": int(row.graph_seed),
            "invasion_total_conductance": metrics["invasion_total_conductance"],
            "invasion_kirchhoff": metrics["invasion_kirchhoff"],
            "invasion_metric": metrics["invasion_metric"],
            "theorem_bound_factor": metrics["theorem_bound_factor"],
            "theorem_rhs": metrics["theorem_rhs"],
        })

    metrics_df = pd.DataFrame(rows)

    # Verify row identity before merging.
    key = ["graph", "n", "graph_seed"]
    if metrics_df.duplicated(key).any():
        raise ValueError("Duplicate graph configuration keys in regenerated metrics.")

    merged = df.merge(metrics_df, on=key, how="left", validate="one_to_one")

    if merged["invasion_metric"].isna().any():
        raise ValueError("Some configurations did not receive weighted metrics.")

    merged.to_csv(OUTDIR / "dual_metric_results.csv", index=False)

    predictors = [
        "lambda2",
        "kirchhoff",
        "invasion_kirchhoff",
        "invasion_metric",
    ]
    available = [p for p in predictors if p in merged.columns]
    comparison = pd.DataFrame(
        [fit_log_model(merged, p) for p in available]
    )
    comparison.to_csv(OUTDIR / "metric_comparison.csv", index=False)

    # Combined models requested for direct comparison.
    valid = (
        (merged["mean_fixation"] > 0)
        & (merged["lambda2"] > 0)
        & (merged["kirchhoff"] > 0)
        & (merged["invasion_metric"] > 0)
    )
    d = merged.loc[valid].copy()

    y = np.log(d["mean_fixation"].to_numpy())

    X_old = sm.add_constant(
        np.column_stack([
            np.log(d["kirchhoff"].to_numpy()),
            np.log(d["lambda2"].to_numpy()),
        ])
    )
    old_model = sm.OLS(y, X_old).fit()

    X_new = sm.add_constant(np.log(d["invasion_metric"].to_numpy()))
    new_model = sm.OLS(y, X_new).fit()

    model_rows = [
        {
            "model": "ordinary_K_plus_lambda2",
            "r2": old_model.rsquared,
            "rmse_log": np.sqrt(np.mean(old_model.resid ** 2)),
            "mae_log": np.mean(np.abs(old_model.resid)),
            "intercept": old_model.params[0],
            "coef_log_K_or_metric": old_model.params[1],
            "coef_log_lambda2": old_model.params[2],
        },
        {
            "model": "invasion_metric_only",
            "r2": new_model.rsquared,
            "rmse_log": np.sqrt(np.mean(new_model.resid ** 2)),
            "mae_log": np.mean(np.abs(new_model.resid)),
            "intercept": new_model.params[0],
            "coef_log_K_or_metric": new_model.params[1],
            "coef_log_lambda2": np.nan,
        },
    ]

    pd.DataFrame(model_rows).to_csv(
        OUTDIR / "model_comparison.csv", index=False
    )

    # Regular-graph identity check. For a d-regular graph:
    # W_I K_I = m K.
    identity_rows = []
    for family, gdf in merged.groupby("graph", sort=True):
        for row in gdf.itertuples(index=False):
            G = generate_graph(row.graph, int(row.n), seed=int(row.graph_seed))
            degrees = [d for _, d in G.degree()]
            if len(set(degrees)) == 1:
                ordinary = float(row.kirchhoff)
                product = float(row.invasion_metric)
                m = G.number_of_edges()
                relerr = abs(product - m * ordinary) / max(1.0, abs(m * ordinary))
                identity_rows.append({
                    "graph": family,
                    "n": int(row.n),
                    "relative_error": relerr,
                })

    identity_df = pd.DataFrame(identity_rows)
    identity_df.to_csv(
        OUTDIR / "regular_graph_identity_check.csv", index=False
    )

    print("=" * 78)
    print("INVASION-PROCESS WEIGHTED METRIC")
    print("=" * 78)
    print(
        merged[
            [
                "graph", "n", "lambda2", "kirchhoff",
                "invasion_total_conductance",
                "invasion_kirchhoff",
                "invasion_metric",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.8g}")
    )

    print("\n" + "=" * 78)
    print("SINGLE-METRIC COMPARISON")
    print("=" * 78)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.8g}"))

    print("\n" + "=" * 78)
    print("MODEL COMPARISON")
    print("=" * 78)
    print(
        pd.DataFrame(model_rows).to_string(
            index=False, float_format=lambda x: f"{x:.8g}"
        )
    )

    if not identity_df.empty:
        print("\n" + "=" * 78)
        print("REGULAR-GRAPH IDENTITY: W_I K_I = m K")
        print("=" * 78)
        print(
            identity_df.to_string(
                index=False, float_format=lambda x: f"{x:.3e}"
            )
        )


if __name__ == "__main__":
    main()
