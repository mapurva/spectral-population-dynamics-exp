from src.graphs import generate_graph


def test_barbell_has_requested_number_of_nodes():
    for n in [20, 30, 50, 80, 120]:
        G = generate_graph("barbell", n)

        assert G.number_of_nodes() == n

def test_erdos_renyi_reproducible_with_seed():
    from src.graphs import generate_graph

    G1 = generate_graph("erdos_renyi", 50, seed=12345)
    G2 = generate_graph("erdos_renyi", 50, seed=12345)

    assert G1.number_of_nodes() == G2.number_of_nodes()
    assert G1.number_of_edges() == G2.number_of_edges()
    assert set(G1.edges()) == set(G2.edges())



def test_barbell_is_connected():
    for n in [20, 30, 50, 80, 120]:
        G = generate_graph("barbell", n)

        assert G.number_of_nodes() == n
        assert G.number_of_edges() == 2 * ((n // 2) * (n // 2 - 1) // 2) + 1