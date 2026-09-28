import math

from src.graphs import generate_graph
from src.resistance import kirchhoff_index


def test_complete_graph_kirchhoff_index():
    for n in [20, 30, 50, 80, 120]:
        G = generate_graph("complete", n)
        expected = n - 1
        actual = kirchhoff_index(G)

        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)


def test_path_graph_kirchhoff_index():
    for n in [20, 30, 50, 80, 120]:
        G = generate_graph("path", n)
        expected = (n**3 - n) / 6
        actual = kirchhoff_index(G)

        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)


def test_cycle_graph_kirchhoff_index():
    for n in [20, 30, 50, 80, 120]:
        G = generate_graph("cycle", n)
        expected = (n**3 - n) / 12
        actual = kirchhoff_index(G)

        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)


def test_star_graph_kirchhoff_index():
    for n in [20, 30, 50, 80, 120]:
        G = generate_graph("star", n)
        expected = (n - 1) ** 2
        actual = kirchhoff_index(G)

        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)