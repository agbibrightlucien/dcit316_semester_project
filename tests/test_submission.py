"""End-to-end integration checks for the influence-maximization pipeline.

Unlike test_functions.py (unit tests on toy graphs), these tests run the
real pipeline against the actual project dataset (data/facebook_combined.txt),
using deliberately small budgets / MC run counts so the suite still runs
quickly. They exist to catch integration breakages between modules, not to
validate statistical properties of the results (that's what
reports/interpretation.md is for).
"""

from pathlib import Path

import pytest

from src.evaluation import estimate_spread
from src.graph_utils import compute_basic_stats, load_graph
from src.seed_selection import degree_ranking, greedy_seeds_incremental, random_seeds

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "facebook_combined.txt"

requires_dataset = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="facebook_combined.txt not present (see data_card.md)"
)


@pytest.fixture(scope="module")
def graph():
    return load_graph(DATA_PATH)


@requires_dataset
class TestDatasetIntegrity:
    def test_expected_scale(self, graph):
        # SNAP ego-Facebook: 4,039 nodes, 88,234 edges (see data_card.md).
        assert graph.number_of_nodes() == 4039
        assert graph.number_of_edges() == 88234

    def test_graph_is_undirected_and_simple(self, graph):
        assert not graph.is_directed()
        assert not any(graph.has_edge(n, n) for n in list(graph.nodes())[:50])


@requires_dataset
class TestBasicStatsPipeline:
    def test_stats_are_internally_consistent(self, graph):
        stats = compute_basic_stats(graph)
        assert stats["num_nodes"] == graph.number_of_nodes()
        assert stats["num_edges"] == graph.number_of_edges()
        assert 0.0 <= stats["density"] <= 1.0
        assert stats["largest_component_size"] <= stats["num_nodes"]
        assert stats["num_connected_components"] >= 1


@requires_dataset
class TestEndToEndSmallRun:
    """A cheap, full-pipeline smoke test: load -> select seeds -> evaluate."""

    def test_greedy_beats_random_at_small_budget(self, graph):
        k = 5
        greedy = greedy_seeds_incremental(graph, model="IC", max_k=k, mc_runs=50, seed=42, p=0.01)
        greedy_seeds = greedy["order"]
        rand_seeds = random_seeds(graph, max_k=k, seed=42)

        greedy_spread = estimate_spread(graph, greedy_seeds, model="IC", mc_runs=100, seed=1, p=0.01)
        random_spread = estimate_spread(graph, rand_seeds, model="IC", mc_runs=100, seed=1, p=0.01)

        # Greedy is provably near-optimal for submodular spread maximization,
        # so on a graph this size it should not be beaten by uniform random
        # seeding at a small budget.
        assert greedy_spread >= random_spread

    def test_degree_heuristic_produces_valid_seed_set(self, graph):
        ranking = degree_ranking(graph)
        top5 = ranking[:5]
        assert len(top5) == 5
        assert all(node in graph for node in top5)

    def test_spread_is_monotone_nondecreasing_in_budget(self, graph):
        greedy = greedy_seeds_incremental(graph, model="IC", max_k=20, mc_runs=50, seed=42, p=0.01)
        spreads = greedy["cumulative_spread"]
        assert spreads[4] <= spreads[9] <= spreads[19]  # k=5 <= k=10 <= k=20


@requires_dataset
class TestReproducibility:
    def test_full_small_pipeline_is_reproducible(self, graph):
        run_a = greedy_seeds_incremental(graph, model="LT", max_k=5, mc_runs=40, seed=7)
        run_b = greedy_seeds_incremental(graph, model="LT", max_k=5, mc_runs=40, seed=7)
        assert run_a["order"] == run_b["order"]
        assert run_a["cumulative_spread"] == run_b["cumulative_spread"]
