from src.seeding import configuration_seed


def test_seed_is_reproducible():
    seed1 = configuration_seed(
        20260928,
        "erdos_renyi",
        50,
        "graph"
    )

    seed2 = configuration_seed(
        20260928,
        "erdos_renyi",
        50,
        "graph"
    )

    assert seed1 == seed2


def test_graph_and_simulation_streams_are_distinct():
    graph_seed = configuration_seed(
        20260928,
        "erdos_renyi",
        50,
        "graph"
    )

    simulation_seed = configuration_seed(
        20260928,
        "erdos_renyi",
        50,
        "simulation"
    )

    assert graph_seed != simulation_seed


def test_different_configurations_have_distinct_seeds():
    seed1 = configuration_seed(
        20260928,
        "cycle",
        20,
        "graph"
    )

    seed2 = configuration_seed(
        20260928,
        "cycle",
        30,
        "graph"
    )

    assert seed1 != seed2