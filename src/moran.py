import random


def run_moran_process(G, max_steps=50000, rng=None):
    """
    Run one neutral Moran/voter-style fixation process.

    Parameters
    ----------
    G : networkx.Graph
        Connected population graph.
    max_steps : int
        Maximum number of update steps. If fixation is not reached,
        the run is right-censored at max_steps.
    rng : random.Random, optional
        Dedicated random-number generator for reproducibility.

    Returns
    -------
    int or None
        Fixation time in update steps, or None if the run is censored.
    """
    if rng is None:
        rng = random.Random()

    nodes = list(G.nodes())

    # Initial mutant
    states = {node: 0 for node in nodes}
    mutant = rng.choice(nodes)
    states[mutant] = 1

    for t in range(max_steps):

        # Pick reproducing node uniformly
        parent = rng.choice(nodes)

        # Pick neighbor uniformly
        neighbors = list(G.neighbors(parent))
        if not neighbors:
            continue

        child = rng.choice(neighbors)

        # Copy state
        states[child] = states[parent]

        # Check fixation
        total = sum(states.values())

        if total == 0 or total == len(nodes):
            return t

    # Right-censored: fixation not observed within max_steps
    return None