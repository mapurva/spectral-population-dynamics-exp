import networkx as nx

def generate_graph(graph_type, n):
    if graph_type == "cycle":
        return nx.cycle_graph(n)

    elif graph_type == "complete":
        return nx.complete_graph(n)

    elif graph_type == "erdos_renyi":
        # ensure connectivity
        while True:
            G = nx.erdos_renyi_graph(n, p=0.2)
            if nx.is_connected(G):
                return G

    elif graph_type == "path":
        return nx.path_graph(n)

    elif graph_type == "star":
        return nx.star_graph(n - 1)

    elif graph_type == "barbell":
        return nx.barbell_graph(n // 2, 1)

    elif graph_type == "grid":
        side = int(n**0.5)
        G = nx.grid_2d_graph(side, side)
        return nx.convert_node_labels_to_integers(G)

    else:
        raise ValueError("Unknown graph type")