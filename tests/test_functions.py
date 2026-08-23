"""Unit tests for diffusion models, seed selection, and evaluation logic.

These run on small toy graphs (not the full Facebook dataset) so they are
fast and their expected outcomes can be reasoned about by hand.
"""

import networkx as nx
import numpy as np
import pytest

from src.diffusion_models import (
    reachable_from,
    sample_live_edge_graph_ic,
    sample_live_edge_graph_lt,
    simulate_ic,
    simulate_lt,
)
from src.evaluation import estimate_spread, estimate_spread_from_live_graphs, generate_live_edge_graphs
from src.graph_utils import compute_basic_stats, load_graph
from src.seed_selection import (
    betweenness_ranking,
    degree_ranking,
    eigenvector_ranking,
    greedy_seeds_incremental,
    pagerank_ranking,
    random_seeds,
    ranking_to_seed_sets,
)


def path_graph():
    # 0 - 1 - 2 - 3 - 4
    return nx.path_graph(5)


def star_graph():
    # node 0 connected to 1..6
    return nx.star_graph(6)


class TestReachability:
    def test_reachable_from_basic(self):
        adjacency = {0: [1, 2], 1: [3], 2: [], 3: []}
        assert reachable_from(adjacency, [0]) == {0, 1, 2, 3}

    def test_reachable_from_isolated_seed(self):
        adjacency = {0: [1]}
        assert reachable_from(adjacency, [5]) == {5}

    def test_reachable_from_empty_seeds(self):
        adjacency = {0: [1]}
        assert reachable_from(adjacency, []) == set()


class TestIndependentCascade:
    def test_p_zero_no_spread(self):
        g = path_graph()
        rng = np.random.default_rng(0)
        activated = simulate_ic(g, [0], p=0.0, rng=rng)
        assert activated == {0}

    def test_p_one_full_spread_on_connected_graph(self):
        g = path_graph()
        rng = np.random.default_rng(0)
        activated = simulate_ic(g, [0], p=1.0, rng=rng)
        assert activated == set(g.nodes())

    def test_live_edge_graph_matches_direct_simulation(self):
        g = star_graph()
        rng = np.random.default_rng(1)
        live_graph = sample_live_edge_graph_ic(g, p=1.0, rng=rng)
        assert reachable_from(live_graph, [0]) == set(g.nodes())

    def test_reproducible_with_fixed_seed(self):
        g = nx.erdos_renyi_graph(30, 0.1, seed=7)
        seeds = [0, 1]
        a = simulate_ic(g, seeds, p=0.1, rng=np.random.default_rng(123))
        b = simulate_ic(g, seeds, p=0.1, rng=np.random.default_rng(123))
        assert a == b


class TestLinearThreshold:
    def test_isolated_node_never_activates(self):
        g = nx.Graph()
        g.add_node(0)
        g.add_edge(1, 2)
        rng = np.random.default_rng(0)
        activated = simulate_lt(g, [1], rng=rng)
        assert 0 not in activated

    def test_all_nodes_seeded_means_all_activated(self):
        g = star_graph()
        rng = np.random.default_rng(0)
        activated = simulate_lt(g, list(g.nodes()), rng=rng)
        assert activated == set(g.nodes())

    def test_star_center_seed_can_reach_leaves(self):
        # Center has degree 6; every leaf's only neighbor is the center, so
        # each leaf's single live in-arc must come from the center.
        g = star_graph()
        rng = np.random.default_rng(0)
        activated = simulate_lt(g, [0], rng=rng)
        assert activated == set(g.nodes())


class TestEvaluation:
    def test_estimate_spread_of_empty_seed_set_is_zero(self):
        g = path_graph()
        assert estimate_spread(g, [], model="IC", mc_runs=10, seed=1) == 0.0

    def test_estimate_spread_matches_manual_average(self):
        g = star_graph()
        live_graphs = generate_live_edge_graphs(g, "IC", num_samples=20, seed=5, p=1.0)
        est = estimate_spread_from_live_graphs(live_graphs, [0])
        assert est == pytest.approx(7.0)  # p=1.0 -> every simulation reaches all 7 nodes

    def test_estimate_spread_reproducible(self):
        g = nx.erdos_renyi_graph(25, 0.15, seed=3)
        a = estimate_spread(g, [0, 1], model="LT", mc_runs=30, seed=99)
        b = estimate_spread(g, [0, 1], model="LT", mc_runs=30, seed=99)
        assert a == b


class TestSeedSelection:
    def test_random_seeds_reproducible_and_correct_size(self):
        g = nx.erdos_renyi_graph(50, 0.1, seed=2)
        a = random_seeds(g, max_k=10, seed=42)
        b = random_seeds(g, max_k=10, seed=42)
        assert a == b
        assert len(a) == 10
        assert len(set(a)) == 10  # no duplicates

    def test_degree_ranking_orders_by_degree(self):
        g = star_graph()
        ranking = degree_ranking(g)
        assert ranking[0] == 0  # center has highest degree

    def test_pagerank_ranking_covers_all_nodes(self):
        g = nx.erdos_renyi_graph(20, 0.2, seed=4)
        ranking = pagerank_ranking(g)
        assert set(ranking) == set(g.nodes())

    def test_eigenvector_ranking_covers_all_nodes(self):
        g = nx.erdos_renyi_graph(20, 0.3, seed=4)
        ranking = eigenvector_ranking(g)
        assert set(ranking) == set(g.nodes())

    def test_betweenness_ranking_star_center_highest(self):
        g = star_graph()
        ranking = betweenness_ranking(g, seed=1)
        assert ranking[0] == 0

    def test_ranking_to_seed_sets_are_nested_prefixes(self):
        ranking = [10, 20, 30, 40, 50]
        seed_sets = ranking_to_seed_sets(ranking, budgets=[2, 4])
        assert seed_sets[2] == [10, 20]
        assert seed_sets[4] == [10, 20, 30, 40]
        assert seed_sets[2] == seed_sets[4][:2]

    def test_greedy_seeds_no_duplicates_and_correct_length(self):
        g = nx.erdos_renyi_graph(40, 0.1, seed=6)
        result = greedy_seeds_incremental(g, model="IC", max_k=8, mc_runs=30, seed=42, p=0.1)
        assert len(result["order"]) == 8
        assert len(set(result["order"])) == 8

    def test_greedy_is_monotonically_nondecreasing_spread(self):
        g = nx.erdos_renyi_graph(40, 0.1, seed=6)
        result = greedy_seeds_incremental(g, model="IC", max_k=8, mc_runs=30, seed=42, p=0.1)
        spreads = result["cumulative_spread"]
        assert all(spreads[i] <= spreads[i + 1] + 1e-9 for i in range(len(spreads) - 1))

    def test_greedy_prefix_property(self):
        """Greedy's top-k picks for a small budget equal the prefix of a larger run."""
        g = nx.erdos_renyi_graph(40, 0.15, seed=9)
        full = greedy_seeds_incremental(g, model="IC", max_k=10, mc_runs=25, seed=1, p=0.1)
        small = greedy_seeds_incremental(g, model="IC", max_k=4, mc_runs=25, seed=1, p=0.1)
        assert small["order"] == full["order"][:4]

    def test_greedy_reproducible_with_fixed_seed(self):
        g = nx.erdos_renyi_graph(30, 0.15, seed=11)
        a = greedy_seeds_incremental(g, model="LT", max_k=5, mc_runs=20, seed=42)
        b = greedy_seeds_incremental(g, model="LT", max_k=5, mc_runs=20, seed=42)
        assert a["order"] == b["order"]


class TestGraphUtils:
    def test_load_graph_toy_edgelist(self, tmp_path):
        edge_file = tmp_path / "toy.txt"
        edge_file.write_text("0 1\n1 2\n2 0\n")
        g = load_graph(edge_file)
        assert g.number_of_nodes() == 3
        assert g.number_of_edges() == 3

    def test_load_graph_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_graph(tmp_path / "does_not_exist.txt")

    def test_compute_basic_stats_triangle(self):
        g = nx.complete_graph(3)
        stats = compute_basic_stats(g)
        assert stats["num_nodes"] == 3
        assert stats["num_edges"] == 3
        assert stats["avg_degree"] == pytest.approx(2.0)
        assert stats["num_connected_components"] == 1
        assert stats["average_clustering_coefficient"] == pytest.approx(1.0)
