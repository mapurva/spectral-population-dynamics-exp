import pandas as pd

from src.graphs import generate_graph
from src.spectral import spectral_gap
from src.simulation import estimate_fixation_time
from src.structure import compute_structure_metrics
from src.resistance import kirchhoff_index
from src.seeding import configuration_seed


# ============================================================
# Experiment configuration
# ============================================================

GRAPH_TYPES = [
    "cycle",
    "complete",
    "erdos_renyi",
    "path",
    "star",
    "barbell",
]

SIZES = [20, 30, 50, 80, 120]

TRIALS = 500
MAX_STEPS = 50000

MASTER_SEED = 20260928


def main():
    # ========================================================
    # Run experiment
    # ========================================================

    results = []

    for gtype in GRAPH_TYPES:
        for n in SIZES:

            graph_seed = configuration_seed(
                MASTER_SEED,
                gtype,
                n,
                "graph"
            )

            simulation_seed = configuration_seed(
                MASTER_SEED,
                gtype,
                n,
                "simulation"
            )

            print(
                f"Running {gtype}, n={n}, "
                f"graph_seed={graph_seed}, "
                f"simulation_seed={simulation_seed}, "
                f"trials={TRIALS}, "
                f"max_steps={MAX_STEPS}"
            )

            # ------------------------------------------------
            # Graph generation
            # ------------------------------------------------

            G = generate_graph(
                gtype,
                n,
                seed=graph_seed
            )

            if len(G.nodes()) == 0:
                print(
                    f"Skipping {gtype}, n={n}: empty graph"
                )
                continue

            # ------------------------------------------------
            # Spectral
            # ------------------------------------------------

            lambda2 = spectral_gap(G)

            if lambda2 is None:
                print(
                    f"Skipping {gtype}, n={n}: "
                    f"spectral gap unavailable"
                )
                continue

            # ------------------------------------------------
            # Structural metrics
            # ------------------------------------------------

            structure = compute_structure_metrics(G)

            # ------------------------------------------------
            # Kirchhoff index
            # ------------------------------------------------

            try:
                kirchhoff = kirchhoff_index(G)

            except Exception as e:
                print(
                    f"Kirchhoff failed for "
                    f"{gtype}, n={n}: {e}"
                )
                kirchhoff = None

            # ------------------------------------------------
            # Moran simulation
            # ------------------------------------------------

            stats = estimate_fixation_time(
                G,
                trials=TRIALS,
                seed=simulation_seed,
                max_steps=MAX_STEPS,
            )

            # ------------------------------------------------
            # Store results
            # ------------------------------------------------

            results.append({
                "graph": gtype,
                "n": n,
                "lambda2": lambda2,

                "mean_fixation": stats["mean"],
                "std": stats["std"],

                "success": stats["success"],
                "censored": stats["censored"],
                "trials": stats["trials"],
                "max_steps": stats["max_steps"],

                "graph_seed": graph_seed,
                "simulation_seed": simulation_seed,

                "diameter": structure["diameter"],
                "avg_path": structure["avg_path"],
                "degree_var": structure["degree_var"],

                "kirchhoff": kirchhoff,
            })

    # ========================================================
    # Save results
    # ========================================================

    df = pd.DataFrame(results)

    output_path = "data/results_revision.csv"

    df.to_csv(
        output_path,
        index=False
    )

    print()
    print("Experiment complete.")
    print(f"Configurations: {len(df)}")
    print(f"Trials per configuration: {TRIALS}")
    print(f"Maximum steps per trial: {MAX_STEPS}")
    print(f"Master seed: {MASTER_SEED}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()