import networkx as nx

def generate_graph(graph_type, n):
    if graph_type == "cycle":
        return nx.cycle_graph(n)

    elif graph_type == "complete":
        return nx.complete_graph(n)

    elif graph_type == "erdos_renyi":
        return nx.erdos_renyi_graph(n, p=0.2)

    elif graph_type == "path":
        return nx.path_graph(n)

    elif graph_type == "star":
        return nx.star_graph(n - 1)

    elif graph_type == "barbell":
        return nx.barbell_graph(n // 2, 1)

    elif graph_type == "grid":
        side = int(n**0.5)
        return nx.grid_2d_graph(side, side)

    else:
        raise ValueError("Unknown graph type")