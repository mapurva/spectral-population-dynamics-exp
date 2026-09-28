import pandas as pd

from src.graphs import generate_graph
from src.spectral import spectral_gap
from src.simulation import estimate_fixation_time
from src.structure import compute_structure_metrics
from src.resistance import kirchhoff_index   # ✅ NEW IMPORT

graph_types = ["cycle", "complete", "erdos_renyi", "path", "star", "barbell"]
sizes = [20, 30, 50, 80, 120]

results = []

for gtype in graph_types:
    for n in sizes:
        print(f"Running {gtype}, n={n}")

        G = generate_graph(gtype, n)

        if len(G.nodes()) == 0:
            continue

        # --- Spectral ---
        lambda2 = spectral_gap(G)
        if lambda2 is None:
            continue

        # --- Structure ---
        structure = compute_structure_metrics(G)

        # --- Resistance (NEW) ---
        try:
            kirchhoff = kirchhoff_index(G)
        except Exception as e:
            print(f"Kirchhoff failed for {gtype}, n={n}: {e}")
            kirchhoff = None

        # --- Moran Simulation ---
        stats = estimate_fixation_time(G)

        results.append({
            "graph": gtype,
            "n": n,
            "lambda2": lambda2,
            "mean_fixation": stats["mean"],
            "std": stats["std"],
            "success": stats["success"],
            "fail": stats["fail"],
            "diameter": structure["diameter"],
            "avg_path": structure["avg_path"],
            "degree_var": structure["degree_var"],
            "kirchhoff": kirchhoff   # ✅ NEW FIELD
        })

# Save results
df = pd.DataFrame(results)
df.to_csv("data/results_with_resistance.csv", index=False)

print("Done. Results saved to results_with_resistance.csv")