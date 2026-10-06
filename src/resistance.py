"""
Electrical-network metrics for the invasion-process dual.
"""

import math

import networkx as nx
import numpy as np


def kirchhoff_index(G):
    """Return the ordinary unit-conductance Kirchhoff index K(G)."""
    if not nx.is_connected(G):
        raise ValueError("Kirchhoff index requires a connected graph.")

    L = nx.laplacian_matrix(G).astype(float).toarray()
    eigenvalues = np.linalg.eigvalsh(L)
    nonzero = eigenvalues[1:]

    if np.any(nonzero <= 0):
        raise ValueError("Graph Laplacian is not positive definite off the zero mode.")

    n = G.number_of_nodes()
    return float(n * np.sum(1.0 / nonzero))


def invasion_conductance(G, u, v):
    """Degree-weighted conductance c_uv = 1 / (n d_u d_v)."""
    n = G.number_of_nodes()
    du = G.degree[u]
    dv = G.degree[v]

    if du <= 0 or dv <= 0:
        raise ValueError("Invasion-process conductance requires positive endpoint degrees.")

    return 1.0 / (n * du * dv)


def invasion_weighted_laplacian(G):
    """
    Construct the weighted Laplacian explicitly.

    NetworkX's laplacian_matrix(weight=...) expects an edge-attribute
    name rather than a callable, so the degree-dependent conductances
    are assembled explicitly here.
    """
    if not nx.is_connected(G):
        raise ValueError("Weighted Laplacian requires a connected graph.")

    nodes = list(G.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    L = np.zeros((n, n), dtype=float)

    for u, v in G.edges():
        c = invasion_conductance(G, u, v)
        i = index[u]
        j = index[v]
        L[i, i] += c
        L[j, j] += c
        L[i, j] -= c
        L[j, i] -= c

    return L


def invasion_total_conductance(G):
    """Return W_I = sum_{edges} 1/(n d_u d_v)."""
    if not nx.is_connected(G):
        raise ValueError("Weighted metric requires a connected graph.")

    return float(
        sum(invasion_conductance(G, u, v) for u, v in G.edges())
    )


def invasion_kirchhoff_index(G):
    """
    Return K_I(G) for the degree-weighted invasion-process dual.

    K_I = n * trace(L_I^+), where L_I is the weighted Laplacian with
    c_uv = 1/(n d_u d_v).
    """
    L = invasion_weighted_laplacian(G)
    eigenvalues = np.linalg.eigvalsh(L)

    tol = max(1e-12, np.max(np.abs(eigenvalues)) * 1e-12)
    positive = eigenvalues[eigenvalues > tol]

    n = G.number_of_nodes()
    if len(positive) != n - 1:
        raise ValueError(
            f"Unexpected weighted Laplacian spectrum: expected {n-1} "
            f"positive eigenvalues, found {len(positive)}."
        )

    return float(n * np.sum(1.0 / positive))


def invasion_metric(G):
    """Return W_I, K_I, W_I*K_I, and the theorem RHS."""
    n = G.number_of_nodes()
    W_I = invasion_total_conductance(G)
    K_I = invasion_kirchhoff_index(G)
    product = W_I * K_I
    factor = 2.0 * math.e * (math.log(n) + 2.0)

    return {
        "invasion_total_conductance": W_I,
        "invasion_kirchhoff": K_I,
        "invasion_metric": product,
        "theorem_bound_factor": factor,
        "theorem_rhs": factor * product,
    }
