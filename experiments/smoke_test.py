import pandas as pd

from src.graphs import generate_graph
from src.spectral import spectral_gap
from src.simulation import estimate_fixation_time
from src.structure import compute_structure_metrics
from src.resistance import kirchhoff_index


GRAPH_TYPES = [
    "cycle",
    "complete",
    "erdos_renyi",
    "path",
    "star",
    "barbell",
]

SIZES = [20, 30]

TRIALS = 10
MAX_STEPS = 5000

MASTER_SEED = 20260928


def configuration_seed(graph_type, n):
    text = f"{MASTER_SEED}:{graph_type}:{n}"

    seed = 0
    for char in text:
        seed = (seed * 131 + ord(char)) % (2**63)

    return seed


results = []

for gtype in GRAPH_TYPES:
    for n in SIZES:

        seed = configuration_seed(gtype, n)

        print(
            f"Running {gtype}, n={n}, "
            f"seed={seed}"
        )

        G = generate_graph(
            gtype,
            n,
            seed=seed
        )

        lambda2 = spectral_gap(G)
        structure = compute_structure_metrics(G)
        kirchhoff = kirchhoff_index(G)

        stats = estimate_fixation_time(
            G,
            trials=TRIALS,
            seed=seed,
            max_steps=MAX_STEPS,
        )

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


df = pd.DataFrame(results)

output_path = "data/smoke_test_results.csv"
df.to_csv(output_path, index=False)

print()
print("Smoke test complete.")
print(df)
print()
print(f"Saved to: {output_path}")