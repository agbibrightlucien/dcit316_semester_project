"""Seed selection methods compared in my analysis.

- greedy_seeds: the KKT (Kempe, Kleinberg & Tardos 2003) greedy algorithm,
  accelerated with the CELF (Cost-Effective Lazy Forward) lazy-evaluation
  optimization of Leskovec et al. (2007), "Cost-effective Outbreak Detection
  in Networks" (KDD '07). CELF provably returns the *identical* seed set as
  naive greedy (it exploits submodularity to skip marginal-gain
  recomputations that cannot change the next pick), it is a performance
  optimization, not an approximation. This is what makes running true
  greedy, with Monte Carlo spread estimation, tractable on a ~4,000-node
  graph within my timeline.
- Greedy is also *incremental*: because it adds one node at a time in
  strictly decreasing marginal-gain order, the first k nodes it picks when
  targeting budget k=50 are exactly the seed set it would have picked if
  run directly for any smaller budget k' < 50 (same random live-edge
  graphs). So greedy is run once up to the largest budget, and smaller
  budgets are read off as prefixes. This is a correctness-preserving
  performance optimization, not a shortcut around the algorithm.
- Centrality heuristics (degree, PageRank, betweenness, eigenvector) each
  compute one ranking of all nodes; top-k seed sets for every budget are
  read off as prefixes of that single ranking, for the same reason.
- Betweenness centrality: exact betweenness (Brandes' algorithm) was timed
  at ~86 seconds on this graph (4,039 nodes / 88,234 edges), which was
  affordable, so I computed it exactly rather than via sampling. The
  ``sample_size`` parameter is kept so a sampled (approximate) estimator can
  be substituted if I ever re-run this on a much larger graph.
"""

from __future__ import annotations

import heapq
import time

import networkx as nx
import numpy as np

from src.diffusion_models import reachable_from
from src.evaluation import generate_live_edge_graphs

BETWEENNESS_SAMPLE_SIZE = None  # None = exact betweenness (affordable on this graph, ~86s)


def random_seeds(graph: nx.Graph, max_k: int, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes())
    rng.shuffle(nodes)
    return nodes[:max_k]


def degree_ranking(graph: nx.Graph) -> list:
    return [n for n, _ in sorted(graph.degree(), key=lambda x: x[1], reverse=True)]


def pagerank_ranking(graph: nx.Graph) -> list:
    # nx.pagerank's power-iteration is deterministic (no seed parameter exists).
    scores = nx.pagerank(graph)
    return [n for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def betweenness_ranking(graph: nx.Graph, seed: int = 42, sample_size: int | None = BETWEENNESS_SAMPLE_SIZE) -> list:
    k = None if sample_size is None else min(sample_size, graph.number_of_nodes())
    scores = nx.betweenness_centrality(graph, k=k, seed=seed, normalized=True)
    return [n for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def eigenvector_ranking(graph: nx.Graph) -> list:
    scores = nx.eigenvector_centrality(graph, max_iter=1000)
    return [n for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def ranking_to_seed_sets(ranking: list, budgets: list) -> dict:
    """Slice a single node ranking into top-k seed sets for each budget."""
    return {k: ranking[:k] for k in budgets}


def greedy_seeds_incremental(
    graph: nx.Graph,
    model: str,
    max_k: int,
    mc_runs: int,
    seed: int = 42,
    p: float = 0.01,
):
    """Run CELF-accelerated greedy up to ``max_k`` seeds.

    Returns a dict with:
      - 'order': list of nodes in the order greedy selected them (length max_k)
      - 'cumulative_spread': cumulative_spread[i] = MC-estimated spread of order[:i+1]
        (estimated on the same fixed live-edge graphs used for selection)
      - 'cumulative_seconds': cumulative_seconds[i] = wall-clock seconds elapsed
        from the start of the run until the (i+1)-th seed was selected
    """
    start = time.perf_counter()
    live_graphs = generate_live_edge_graphs(graph, model, mc_runs, seed, p=p)
    num_graphs = len(live_graphs)

    # Cache each node's single-source reachable set per live-edge graph.
    # Reachability is monotone under set union of sources, so this cache is
    # all that is ever needed to score any candidate against any partial
    # seed set, no repeated cascade simulation during selection.
    nodes = list(graph.nodes())
    reach_cache = {v: [reachable_from(lg, [v]) for lg in live_graphs] for v in nodes}

    reached = [set() for _ in range(num_graphs)]
    last_updated = {v: 0 for v in nodes}

    def current_gain(v):
        return sum(len(reach_cache[v][g] - reached[g]) for g in range(num_graphs)) / num_graphs

    heap = [(-current_gain(v), v) for v in nodes]
    heapq.heapify(heap)

    order = []
    cumulative_spread = []
    cumulative_seconds = []
    iteration = 0
    total_spread = 0.0

    while len(order) < max_k and heap:
        neg_gain, v = heapq.heappop(heap)
        if last_updated[v] == iteration:
            order.append(v)
            total_spread += -neg_gain
            for g in range(num_graphs):
                reached[g] |= reach_cache[v][g]
            iteration += 1
            cumulative_spread.append(total_spread)
            cumulative_seconds.append(time.perf_counter() - start)
        else:
            new_gain = current_gain(v)
            last_updated[v] = iteration
            heapq.heappush(heap, (-new_gain, v))

    return {
        "order": order,
        "cumulative_spread": cumulative_spread,
        "cumulative_seconds": cumulative_seconds,
    }
