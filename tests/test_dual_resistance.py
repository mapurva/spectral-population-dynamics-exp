import math

import networkx as nx
import pytest

from src.resistance import (
    invasion_kirchhoff_index,
    invasion_metric,
    invasion_total_conductance,
    kirchhoff_index,
)


@pytest.mark.parametrize(
    "G",
    [
        nx.cycle_graph(6),
        nx.complete_graph(8),
        nx.complete_graph(10),
    ],
)
def test_invasion_metric_reduces_to_mk_on_connected_regular_graphs(G):
    ordinary = kirchhoff_index(G)
    metrics = invasion_metric(G)
    m = G.number_of_edges()

    assert math.isclose(
        metrics["invasion_metric"],
        m * ordinary,
        rel_tol=1e-10,
        abs_tol=1e-10,
    )


def test_star_has_positive_weighted_metric():
    G = nx.star_graph(7)

    W = invasion_total_conductance(G)
    K = invasion_kirchhoff_index(G)
    metrics = invasion_metric(G)

    assert W > 0
    assert K > 0
    assert math.isclose(metrics["invasion_metric"], W * K, rel_tol=1e-12)


def test_weighted_metric_rejects_disconnected_graph():
    G = nx.disjoint_union(nx.path_graph(3), nx.path_graph(2))

    with pytest.raises(ValueError):
        invasion_total_conductance(G)

    with pytest.raises(ValueError):
        invasion_kirchhoff_index(G)
