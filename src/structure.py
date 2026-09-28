import numpy as np
import networkx as nx

def compute_structure_metrics(G):
    degrees = [d for _, d in G.degree()]

    degree_var = np.var(degrees)

    try:
        diameter = nx.diameter(G)
    except:
        diameter = None

    try:
        avg_path = nx.average_shortest_path_length(G)
    except:
        avg_path = None

    return {
        "diameter": diameter,
        "avg_path": avg_path,
        "degree_var": degree_var
    }