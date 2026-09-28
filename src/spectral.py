import numpy as np
import networkx as nx

def spectral_gap(G):
    if not nx.is_connected(G):
        return None

    L = nx.laplacian_matrix(G).astype(float)
    eigenvalues = np.linalg.eigvalsh(L.toarray())
    eigenvalues.sort()

    return eigenvalues[1]