import random
import pandas as pd

from src.graphs import generate_graph
from src.spectral import spectral_gap
from src.simulation import estimate_fixation_time
from src.structure import compute_structure_metrics
from src.resistance import kirchhoff_index


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


def configuration_seed(graph_type, n):
    """
    Generate a deterministic seed for one graph configuration.
    """
    text = f"{MASTER_SEED}:{graph_type}:{n}"

    # Stable deterministic seed independent of Python's hash randomization.
    seed = 0
    for char in text:
        seed = (seed * 131 + ord(char)) % (2**63)

    return seed


# ============================================================
# Run experiment
# ============================================================

results = []

for gtype in GRAPH_TYPES:
    for n in SIZES:

        seed = configuration_seed(gtype, n)

        print(
            f"Running {gtype}, n={n}, "
            f"seed={seed}, trials={TRIALS}, max_steps={MAX_STEPS}"
        )

        # ----------------------------------------------------
        # Graph generation
        # ----------------------------------------------------
        G = generate_graph(gtype, n, seed=seed)

        if len(G.nodes()) == 0:
            print(f"Skipping {gtype}, n={n}: empty graph")
            continue

        # ----------------------------------------------------
        # Spectral
        # ----------------------------------------------------
        lambda2 = spectral_gap(G)

        if lambda2 is None:
            print(f"Skipping {gtype}, n={n}: spectral gap unavailable")
            continue

        # ----------------------------------------------------
        # Structural metrics
        # ----------------------------------------------------
        structure = compute_structure_metrics(G)

        # ----------------------------------------------------
        # Kirchhoff index
        # ----------------------------------------------------
        try:
            kirchhoff = kirchhoff_index(G)
        except Exception as e:
            print(
                f"Kirchhoff failed for {gtype}, n={n}: {e}"
            )
            kirchhoff = None

        # ----------------------------------------------------
        # Moran simulation
        # ----------------------------------------------------
        stats = estimate_fixation_time(
            G,
            trials=TRIALS,
            seed=seed,
            max_steps=MAX_STEPS,
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------
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
            "seed": stats["seed"],

            "diameter": structure["diameter"],
            "avg_path": structure["avg_path"],
            "degree_var": structure["degree_var"],

            "kirchhoff": kirchhoff,
        })


# ============================================================
# Save results
# ============================================================

df = pd.DataFrame(results)

output_path = "data/results_revision.csv"
df.to_csv(output_path, index=False)

print()
print("Experiment complete.")
print(f"Configurations: {len(df)}")
print(f"Trials per configuration: {TRIALS}")
print(f"Maximum steps per trial: {MAX_STEPS}")
print(f"Master seed: {MASTER_SEED}")
print(f"Results saved to: {output_path}")