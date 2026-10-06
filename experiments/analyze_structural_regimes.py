"""
Structural/regime-dependence analysis for the revised SPD experiment.

This script uses the already-generated data/results_revision.csv and the
recorded graph seeds. It does NOT rerun simulations.

It evaluates:
  1. Baseline empirical models:
       log(T) ~ log(lambda2)
       log(T) ~ log(K)
       log(T) ~ log(K) + log(lambda2)

  2. Structural descriptors:
       diameter
       average shortest-path length
       degree variance

  3. Augmented models:
       K + lambda2 + diameter
       K + lambda2 + average path length
       K + lambda2 + degree variance

  4. Nested model comparisons using adjusted R^2 and AIC/BIC.

  5. Family-specific K and lambda2 slopes.

  6. Family fixed effects:
       log(T) ~ log(K) + log(lambda2) + C(graph)

  7. Interaction model:
       log(T) ~ log(K) + log(lambda2) + C(graph)
                + log(K):C(graph)

  8. Five-fold shuffled CV for selected models.

  9. Leave-one-family-out CV for selected models.

The goal is diagnostic, not model selection by optimization. The results
should be interpreted in light of the six graph families and only 30
graph configurations.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import networkx as nx
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.graphs import generate_graph


INPUT = ROOT / "data" / "results_revision.csv"
OUTDIR = ROOT / "data" / "revision_structural_analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 20260928


def graph_descriptors(G):
    """Return structural descriptors used in the reviewer diagnostic."""
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    return {
        "diameter": float(nx.diameter(G)),
        "avg_shortest_path": float(nx.average_shortest_path_length(G)),
        "degree_variance": float(np.var(degrees)),
    }

def add_descriptors(df):
    """Regenerate structural descriptors from the recorded graph seeds."""
    rows = []

    for row in df.itertuples(index=False):
        G = generate_graph(
            row.graph,
            int(row.n),
            seed=int(row.graph_seed),
        )

        desc = graph_descriptors(G)

        rows.append({
            "graph": row.graph,
            "n": int(row.n),
            "graph_seed": int(row.graph_seed),
            "diameter": desc["diameter"],
            "avg_shortest_path": desc["avg_shortest_path"],
            "degree_variance": desc["degree_variance"],
        })

    desc_df = pd.DataFrame(rows)

    # Explicitly verify that the source keys are unique.
    key = ["graph", "n", "graph_seed"]

    if df.duplicated(key).any():
        raise ValueError("Duplicate graph/n/graph_seed configurations in input data.")

    if desc_df.duplicated(key).any():
        raise ValueError("Duplicate regenerated graph/n/graph_seed configurations.")

    # Remove any existing descriptor columns before merging, if present.
    df = df.drop(
        columns=[
            "diameter",
            "avg_shortest_path",
            "degree_variance",
        ],
        errors="ignore",
    )

    return df.merge(
        desc_df,
        on=key,
        how="left",
        validate="one_to_one",
    )
	


def prepare(df):
    d = df.copy()
    positive = (
        (d["mean_fixation"] > 0)
        & (d["lambda2"] > 0)
        & (d["kirchhoff"] > 0)
        & (d["diameter"] > 0)
        & (d["avg_shortest_path"] > 0)
        & (d["degree_variance"] >= 0)
    )
    d = d.loc[positive].copy()

    d["log_T"] = np.log(d["mean_fixation"])
    d["log_lambda2"] = np.log(d["lambda2"])
    d["log_K"] = np.log(d["kirchhoff"])
    d["log_diameter"] = np.log(d["diameter"])
    d["log_avg_path"] = np.log(d["avg_shortest_path"])

    # log1p is used only for the descriptor because degree variance can be zero.
    d["log1p_degree_variance"] = np.log1p(d["degree_variance"])

    return d


def fit_formula(d, formula):
    model = smf.ols(formula, data=d).fit()
    return {
        "formula": formula,
        "n": int(model.nobs),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "rmse_log": float(np.sqrt(np.mean(model.resid ** 2))),
        "mae_log": float(np.mean(np.abs(model.resid))),
    }, model


def cv_formula(d, formula, groups=None, folds=5):
    """
    Five-fold CV on log(T). If groups is supplied, this function is not used;
    leave-one-family-out is handled separately.
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    y_all = d["log_T"].to_numpy()
    preds = np.full(len(d), np.nan)

    for train_idx, test_idx in kf.split(d):
        train = d.iloc[train_idx]
        test = d.iloc[test_idx]
        model = smf.ols(formula, data=train).fit()
        preds[test_idx] = model.predict(test)

    valid = np.isfinite(preds)
    return {
        "formula": formula,
        "folds": folds,
        "r2": float(r2_score(y_all[valid], preds[valid])),
        "rmse_log": float(np.sqrt(mean_squared_error(y_all[valid], preds[valid]))),
        "mae_log": float(mean_absolute_error(y_all[valid], preds[valid])),
    }


def leave_one_family_out(d, formula):
    rows = []

    for family in sorted(d["graph"].unique()):
        train = d[d["graph"] != family]
        test = d[d["graph"] == family]

        model = smf.ols(formula, data=train).fit()
        pred = model.predict(test)
        y = test["log_T"]

        rows.append({
            "held_out_family": family,
            "n_test": len(test),
            "r2": float(r2_score(y, pred)),
            "rmse_log": float(np.sqrt(mean_squared_error(y, pred))),
            "mae_log": float(mean_absolute_error(y, pred)),
        })

    return pd.DataFrame(rows)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing {INPUT}")

    df = pd.read_csv(INPUT)
    df = add_descriptors(df)
    d = prepare(df)

    d.to_csv(OUTDIR / "structural_results.csv", index=False)

    formulas = [
        "log_T ~ log_lambda2",
        "log_T ~ log_K",
        "log_T ~ log_K + log_lambda2",
        "log_T ~ log_K + log_lambda2 + log_diameter",
        "log_T ~ log_K + log_lambda2 + log_avg_path",
        "log_T ~ log_K + log_lambda2 + log1p_degree_variance",
        "log_T ~ log_K + log_lambda2 + C(graph)",
        "log_T ~ log_K + log_lambda2 + C(graph) + log_K:C(graph)",
    ]

    fit_rows = []
    fitted = {}

    for formula in formulas:
        row, model = fit_formula(d, formula)
        fit_rows.append(row)
        fitted[formula] = model

    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(OUTDIR / "model_comparison.csv", index=False)

    selected_cv = [
        "log_T ~ log_lambda2",
        "log_T ~ log_K",
        "log_T ~ log_K + log_lambda2",
        "log_T ~ log_K + log_lambda2 + log_diameter",
        "log_T ~ log_K + log_lambda2 + log1p_degree_variance",
    ]

    cv_df = pd.DataFrame([cv_formula(d, f) for f in selected_cv])
    cv_df.to_csv(OUTDIR / "five_fold_cv.csv", index=False)

    loo_formulas = [
        "log_T ~ log_K + log_lambda2",
        "log_T ~ log_K + log_lambda2 + log_diameter",
        "log_T ~ log_K + log_lambda2 + log1p_degree_variance",
    ]

    loo_parts = []
    for formula in loo_formulas:
        tmp = leave_one_family_out(d, formula)
        tmp.insert(0, "formula", formula)
        loo_parts.append(tmp)

    loo_df = pd.concat(loo_parts, ignore_index=True)
    loo_df.to_csv(OUTDIR / "leave_one_family_out.csv", index=False)

    # Family-specific scaling diagnostics.
    family_rows = []
    for family, g in d.groupby("graph", sort=True):
        for predictor in ["log_K", "log_lambda2"]:
            if g[predictor].nunique() < 2:
                family_rows.append({
                    "graph": family,
                    "predictor": predictor,
                    "n": len(g),
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r2": np.nan,
                    "status": "non-identifiable",
                })
                continue

            X = sm.add_constant(g[predictor])
            model = sm.OLS(g["log_T"], X).fit()
            family_rows.append({
                "graph": family,
                "predictor": predictor,
                "n": len(g),
                "slope": float(model.params.iloc[1]),
                "intercept": float(model.params.iloc[0]),
                "r2": float(model.rsquared),
                "status": "identified",
            })

    family_df = pd.DataFrame(family_rows)
    family_df.to_csv(OUTDIR / "family_scaling.csv", index=False)

    # Correlations among graph descriptors and predictors.
    corr_cols = [
        "log_lambda2",
        "log_K",
        "log_diameter",
        "log_avg_path",
        "log1p_degree_variance",
    ]
    corr = d[corr_cols].corr(method="pearson")
    corr.to_csv(OUTDIR / "descriptor_correlations.csv")

    # Print concise but complete diagnostics.
    print("=" * 90)
    print("STRUCTURAL DESCRIPTORS")
    print("=" * 90)
    print(
        d[
            [
                "graph", "n", "lambda2", "kirchhoff",
                "diameter", "avg_shortest_path", "degree_variance",
                "mean_fixation",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.8g}")
    )

    print("\n" + "=" * 90)
    print("IN-SAMPLE MODEL COMPARISON")
    print("=" * 90)
    print(
        fit_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.8g}",
        )
    )

    print("\n" + "=" * 90)
    print("5-FOLD SHUFFLED CV")
    print("=" * 90)
    print(
        cv_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.8g}",
        )
    )

    print("\n" + "=" * 90)
    print("LEAVE-ONE-FAMILY-OUT CV")
    print("=" * 90)
    print(
        loo_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.8g}",
        )
    )

    print("\n" + "=" * 90)
    print("FAMILY-SPECIFIC SCALING")
    print("=" * 90)
    print(
        family_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.8g}",
        )
    )

    print("\n" + "=" * 90)
    print("PREDICTOR / STRUCTURAL CORRELATIONS")
    print("=" * 90)
    print(corr.to_string(float_format=lambda x: f"{x:.6f}"))

    print("\nResults written to:")
    print(OUTDIR)


if __name__ == "__main__":
    main()
