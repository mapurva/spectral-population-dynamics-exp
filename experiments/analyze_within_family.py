"""
Within-family structural diagnostic for the revised SPD experiment.

Purpose:
  - determine whether degree heterogeneity adds information beyond K and
    lambda2 after removing between-family differences;
  - quantify correlations and regressions using family-centered variables;
  - report family-specific residual relationships;
  - correctly mark constant predictors (e.g. star lambda2) as
    non-identifiable.

No simulations are rerun.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.graphs import generate_graph


INPUT = ROOT / "data" / "results_revision.csv"
OUTDIR = ROOT / "data" / "revision_within_family"
OUTDIR.mkdir(parents=True, exist_ok=True)


def graph_descriptors(G):
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    return {
        "diameter": float(nx.diameter(G)),
        "avg_shortest_path": float(nx.average_shortest_path_length(G)),
        "degree_variance": float(np.var(degrees)),
    }


def add_descriptors(df):
    rows = []

    for row in df.itertuples(index=False):
        G = generate_graph(
            row.graph,
            int(row.n),
            seed=int(row.graph_seed),
        )
        rows.append({
            "graph": row.graph,
            "n": int(row.n),
            "graph_seed": int(row.graph_seed),
            **graph_descriptors(G),
        })

    desc = pd.DataFrame(rows)
    key = ["graph", "n", "graph_seed"]

    if df.duplicated(key).any():
        raise ValueError("Duplicate configuration keys in input data.")

    return df.drop(
        columns=["diameter", "avg_shortest_path", "degree_variance"],
        errors="ignore",
    ).merge(
        desc,
        on=key,
        how="left",
        validate="one_to_one",
    )


def prepare(df):
    d = df.copy()

    d["log_T"] = np.log(d["mean_fixation"])
    d["log_K"] = np.log(d["kirchhoff"])
    d["log_lambda2"] = np.log(d["lambda2"])
    d["log1p_degree_variance"] = np.log1p(d["degree_variance"])
    d["log_diameter"] = np.log(d["diameter"])

    # Family-centered variables remove between-family location effects.
    for col in [
        "log_T",
        "log_K",
        "log_lambda2",
        "log1p_degree_variance",
        "log_diameter",
    ]:
        d[f"{col}_within"] = d[col] - d.groupby("graph")[col].transform("mean")

    return d


def centered_regression(d, predictor, response="log_T_within"):
    x = d[predictor].to_numpy()
    y = d[response].to_numpy()

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3 or np.ptp(x[mask]) <= 1e-12:
        return {
            "predictor": predictor,
            "n": int(mask.sum()),
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "status": "non-identifiable",
        }

    model = sm.OLS(y[mask], sm.add_constant(x[mask])).fit()

    return {
        "predictor": predictor,
        "n": int(mask.sum()),
        "slope": float(model.params[1]),
        "intercept": float(model.params[0]),
        "r2": float(model.rsquared),
        "status": "identified",
    }


def family_regressions(d):
    rows = []

    for family, g in d.groupby("graph", sort=True):
        for predictor in [
            "log_K",
            "log_lambda2",
            "log1p_degree_variance",
            "log_diameter",
        ]:
            x = g[predictor].to_numpy()
            y = g["log_T"].to_numpy()

            if len(x) < 3 or np.ptp(x) <= 1e-12:
                rows.append({
                    "graph": family,
                    "predictor": predictor,
                    "n": len(g),
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r2": np.nan,
                    "status": "non-identifiable",
                })
                continue

            model = sm.OLS(y, sm.add_constant(x)).fit()

            rows.append({
                "graph": family,
                "predictor": predictor,
                "n": len(g),
                "slope": float(model.params[1]),
                "intercept": float(model.params[0]),
                "r2": float(model.rsquared),
                "status": "identified",
            })

    return pd.DataFrame(rows)


def residual_analysis(d):
    """
    Remove K + lambda2 first, then test whether degree variance explains
    the remaining within-family variation.
    """
    # Full pooled baseline residuals.
    baseline = smf.ols(
        "log_T ~ log_K + log_lambda2",
        data=d,
    ).fit()

    d = d.copy()
    d["baseline_residual"] = baseline.resid

    # Family-centered residual: remove each family's mean residual.
    d["baseline_residual_within"] = (
        d["baseline_residual"]
        - d.groupby("graph")["baseline_residual"].transform("mean")
    )

    x = d["log1p_degree_variance_within"].to_numpy()
    y = d["baseline_residual_within"].to_numpy()

    mask = np.isfinite(x) & np.isfinite(y)

    if np.ptp(x[mask]) <= 1e-12:
        result = {
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "status": "non-identifiable",
        }
    else:
        model = sm.OLS(y[mask], sm.add_constant(x[mask])).fit()
        result = {
            "slope": float(model.params[1]),
            "intercept": float(model.params[0]),
            "r2": float(model.rsquared),
            "status": "identified",
        }

    return d, result


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing {INPUT}")

    df = pd.read_csv(INPUT)
    d = prepare(add_descriptors(df))

    d.to_csv(OUTDIR / "within_family_data.csv", index=False)

    # Centered regressions: these are the main diagnostic.
    centered = pd.DataFrame([
        centered_regression(d, "log_K_within"),
        centered_regression(d, "log_lambda2_within"),
        centered_regression(d, "log1p_degree_variance_within"),
        centered_regression(d, "log_diameter_within"),
    ])
    centered.to_csv(OUTDIR / "centered_regressions.csv", index=False)

    family = family_regressions(d)
    family.to_csv(OUTDIR / "family_regressions.csv", index=False)

    d, residual_result = residual_analysis(d)
    d.to_csv(OUTDIR / "within_family_residuals.csv", index=False)

    # A family-fixed-effects baseline and an augmented family-fixed-effects
    # model are reported as explanatory, not out-of-family predictive models.
    fe_baseline = smf.ols(
        "log_T ~ log_K + log_lambda2 + C(graph)",
        data=d,
    ).fit()

    fe_augmented = smf.ols(
        "log_T ~ log_K + log_lambda2 + log1p_degree_variance + C(graph)",
        data=d,
    ).fit()

    fe_rows = pd.DataFrame([
        {
            "model": "K + lambda2 + family fixed effects",
            "r2": fe_baseline.rsquared,
            "adj_r2": fe_baseline.rsquared_adj,
            "aic": fe_baseline.aic,
            "bic": fe_baseline.bic,
            "rmse_log": np.sqrt(np.mean(fe_baseline.resid ** 2)),
            "mae_log": np.mean(np.abs(fe_baseline.resid)),
            "degree_variance_coef": np.nan,
            "degree_variance_pvalue": np.nan,
        },
        {
            "model": "K + lambda2 + degree variance + family fixed effects",
            "r2": fe_augmented.rsquared,
            "adj_r2": fe_augmented.rsquared_adj,
            "aic": fe_augmented.aic,
            "bic": fe_augmented.bic,
            "rmse_log": np.sqrt(np.mean(fe_augmented.resid ** 2)),
            "mae_log": np.mean(np.abs(fe_augmented.resid)),
            "degree_variance_coef": fe_augmented.params.get(
                "log1p_degree_variance", np.nan
            ),
            "degree_variance_pvalue": fe_augmented.pvalues.get(
                "log1p_degree_variance", np.nan
            ),
        },
    ])
    fe_rows.to_csv(OUTDIR / "family_fixed_effects.csv", index=False)

    print("=" * 90)
    print("WITHIN-FAMILY CENTERED REGRESSIONS")
    print("=" * 90)
    print(
        centered.to_string(
            index=False,
            float_format=lambda x: f"{x:.8g}",
        )
    )

    print("\n" + "=" * 90)
    print("FAMILY-SPECIFIC REGRESSIONS")
    print("=" * 90)
    print(
        family.to_string(
            index=False,
            float_format=lambda x: f"{x:.8g}",
        )
    )

    print("\n" + "=" * 90)
    print("RESIDUAL AFTER POOLED K + LAMBDA2")
    print("=" * 90)
    print(
        pd.DataFrame([residual_result]).to_string(
            index=False,
            float_format=lambda x: f"{x:.8g}",
        )
    )

    print("\n" + "=" * 90)
    print("FAMILY FIXED-EFFECT MODELS")
    print("=" * 90)
    print(
        fe_rows.to_string(
            index=False,
            float_format=lambda x: f"{x:.8g}",
        )
    )

    print("\nResults written to:")
    print(OUTDIR)


if __name__ == "__main__":
    main()
