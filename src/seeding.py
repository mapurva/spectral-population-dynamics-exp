def configuration_seed(master_seed, graph_type, n, stream):
    """
    Generate a deterministic seed for a specific random stream.

    Parameters
    ----------
    master_seed : int
        Global experiment seed.
    graph_type : str
        Graph family.
    n : int
        Number of vertices.
    stream : str
        Randomness stream, e.g. "graph" or "simulation".

    Returns
    -------
    int
        Deterministic seed.
    """
    text = f"{master_seed}:{graph_type}:{n}:{stream}"

    seed = 0
    for char in text:
        seed = (seed * 131 + ord(char)) % (2**63)

    return seed