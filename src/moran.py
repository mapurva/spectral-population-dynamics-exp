import random

def run_moran_process(G, max_steps=50000):
    nodes = list(G.nodes())

    # initial mutant
    states = {node: 0 for node in nodes}
    mutant = random.choice(nodes)
    states[mutant] = 1

    for t in range(max_steps):

        # pick reproducing node uniformly
        parent = random.choice(nodes)

        # pick neighbor
        neighbors = list(G.neighbors(parent))
        if not neighbors:
            continue

        child = random.choice(neighbors)

        # copy state
        states[child] = states[parent]

        # check fixation
        total = sum(states.values())

        if total == 0 or total == len(nodes):
            return t

    return None  # did not fix