import networkx as nx


def make_barbell_graph(n):
    """
    Construct a barbell graph with exactly n vertices.

    The graph consists of two complete subgraphs of size n/2
    connected by a single bridge edge.

    Parameters
    ----------
    n : int
        Number of vertices. Must be even and >= 4.

    Returns
    -------
    networkx.Graph
        Connected undirected barbell graph with exactly n vertices.
    """
    if n < 4 or n % 2 != 0:
        raise ValueError("Barbell graph requires an even n >= 4.")

    m = n // 2

    left = nx.complete_graph(m)
    right = nx.complete_graph(range(m, n))

    G = nx.compose(left, right)
    G.add_edge(0, m)

    return G


def generate_graph(graph_type, n, seed=None):
    """
    Generate a graph for the specified graph family and size.

    Parameters
    ----------
    graph_type : str
        Graph family.
    n : int
        Requested number of vertices.
    seed : int, optional
        Random seed used for stochastic graph generation.
    """

    if graph_type == "cycle":
        return nx.cycle_graph(n)

    elif graph_type == "complete":
        return nx.complete_graph(n)

    elif graph_type == "erdos_renyi":
        # Use NetworkX's seed parameter so that graph generation
        # is reproducible.
        #
        # If the generated graph is disconnected, increment the
        # seed deterministically and try again.
        attempt = 0

        while True:
            current_seed = None
            if seed is not None:
                current_seed = seed + attempt

            G = nx.erdos_renyi_graph(
                n,
                p=0.2,
                seed=current_seed
            )

            if nx.is_connected(G):
                return G

            attempt += 1

    elif graph_type == "path":
        return nx.path_graph(n)

    elif graph_type == "star":
        return nx.star_graph(n - 1)

    elif graph_type == "barbell":
        return make_barbell_graph(n)

    elif graph_type == "grid":
        side = int(n**0.5)
        G = nx.grid_2d_graph(side, side)
        return nx.convert_node_labels_to_integers(G)

    else:
        raise ValueError("Unknown graph type")