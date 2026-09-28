import networkx as nx

from src.moran import run_moran_process
from src.simulation import estimate_fixation_time


def test_moran_reproducible_with_seed():
    G = nx.path_graph(10)

    result1 = estimate_fixation_time(
        G,
        trials=20,
        seed=12345,
        max_steps=5000
    )

    result2 = estimate_fixation_time(
        G,
        trials=20,
        seed=12345,
        max_steps=5000
    )

    assert result1 == result2


def test_simulation_counts_are_consistent():
    G = nx.complete_graph(10)

    result = estimate_fixation_time(
        G,
        trials=20,
        seed=12345,
        max_steps=5000
    )

    assert result["trials"] == 20
    assert result["success"] + result["censored"] == 20


def test_moran_rng_reproducible():
    G = nx.path_graph(10)

    import random

    rng1 = random.Random(123)
    rng2 = random.Random(123)

    t1 = run_moran_process(G, max_steps=5000, rng=rng1)
    t2 = run_moran_process(G, max_steps=5000, rng=rng2)

    assert t1 == t2