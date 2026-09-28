import numpy as np
import networkx as nx


def kirchhoff_index(G):
    """
    Compute the Kirchhoff index of a connected undirected graph.

    For a connected graph G with n vertices and Laplacian eigenvalues

        0 = lambda_1 < lambda_2 <= ... <= lambda_n,

    the Kirchhoff index is

        K(G) = n * sum_{i=2}^n 1 / lambda_i.

    Parameters
    ----------
    G : networkx.Graph
        Connected, undirected graph.

    Returns
    -------
    float
        Kirchhoff index K(G).
    """
    if not nx.is_connected(G):
        raise ValueError("Kirchhoff index requires a connected graph.")

    n = G.number_of_nodes()

    L = nx.laplacian_matrix(G).astype(float)
    eigenvalues = np.linalg.eigvalsh(L.toarray())

    # The first eigenvalue is zero for a connected graph.
    nonzero = eigenvalues[1:]

    return float(n * np.sum(1.0 / nonzero))