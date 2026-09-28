import numpy as np
from src.moran import run_moran_process

def estimate_fixation_time(G, trials=500):
    times = []
    failures = 0

    for _ in range(trials):
        t = run_moran_process(G)
        if t is None:
            failures += 1
        else:
            times.append(t)

    return {
        "mean": np.mean(times) if times else None,
        "std": np.std(times) if times else None,
        "success": len(times),
        "fail": failures
    }