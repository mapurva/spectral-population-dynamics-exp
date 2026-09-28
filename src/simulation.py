import random
import numpy as np

from src.moran import run_moran_process


def estimate_fixation_time(G, trials=500, seed=None, max_steps=50000):
    """
    Estimate fixation time over repeated Moran-process trials.

    Parameters
    ----------
    G : networkx.Graph
        Population graph.
    trials : int
        Number of independent simulation trials.
    seed : int, optional
        Seed for reproducible trial generation.
    max_steps : int
        Maximum number of update steps per trial.

    Returns
    -------
    dict
        Summary statistics including successful and censored runs.
    """
    times = []
    censored = 0

    master_rng = random.Random(seed)

    for _ in range(trials):
        # Independent RNG stream for each trial
        trial_seed = master_rng.randrange(0, 2**63)
        trial_rng = random.Random(trial_seed)

        t = run_moran_process(
            G,
            max_steps=max_steps,
            rng=trial_rng
        )

        if t is None:
            censored += 1
        else:
            times.append(t)

    return {
        "mean": np.mean(times) if times else None,
        "std": np.std(times) if times else None,
        "success": len(times),
        "censored": censored,
        "trials": trials,
        "max_steps": max_steps,
        "seed": seed,
    }