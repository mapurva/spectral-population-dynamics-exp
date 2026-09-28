import numpy as np
import networkx as nx

def kirchhoff_index(G):
    L = nx.laplacian_matrix(G).astype(float)
    eigenvalues = np.linalg.eigvalsh(L.toarray())
    
    # avoid zero eigenvalue
    nonzero = eigenvalues[1:]
    
    return np.sum(1.0 / nonzero)