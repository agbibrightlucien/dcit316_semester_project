"""Independent Cascade (IC) and Linear Threshold (LT) diffusion models.

Both models follow Kempe, Kleinberg & Tardos (2003), "Maximizing the Spread
of Influence through a Social Network" (KDD '03).

Implementation note (important for readers of the paper):
Both IC and LT have an equivalent "live-edge graph" formulation (KKT 2003,
Theorem 2.2 / Sec. 4): a single stochastic simulation of either diffusion
process is exactly equivalent to (a) sampling a random subgraph of "live"
edges once, then (b) computing which nodes are reachable from the seed set
in that subgraph via a plain BFS. This module implements diffusion via that
equivalence, because it lets the same sampled live-edge graph be reused
across many marginal-gain evaluations during greedy seed selection
(instead of re-simulating the whole cascade from scratch every time), which
is what makes running the exact KKT greedy algorithm on a ~4,000-node graph
computationally feasible. This is a performance optimization only: the
random process being simulated, and therefore the expected spread it
estimates, is unchanged.

- IC: each undirected edge {u, v} yields two independent directed arcs
  u->v and v->u, each "live" independently with probability p.
- LT: each node v independently picks at most one of its neighbors as its
  sole "live" in-arc, with probability 1 / degree(v) per neighbor (i.e.
  uniform edge weights that sum to 1 across v's in-edges, the standard KKT
  parameterization). This is equivalent to giving v a uniform random
  threshold in [0, 1] and uniform edge weights 1/degree(v).
"""

from __future__ import annotations

from collections import defaultdict, deque

import networkx as nx
import numpy as np

LiveEdgeGraph = dict  # node -> list[node], directed adjacency of "live" arcs


def reachable_from(adjacency: LiveEdgeGraph, seeds) -> set:
    """BFS reachability from a seed set over a directed live-edge adjacency."""
    visited = set(seeds)
    queue = deque(seeds)
    while queue:
        u = queue.popleft()
        for v in adjacency.get(u, ()):
            if v not in visited:
                visited.add(v)
                queue.append(v)
    return visited


def sample_live_edge_graph_ic(graph: nx.Graph, p: float, rng: np.random.Generator) -> LiveEdgeGraph:
    """Sample one live-edge graph under the Independent Cascade model."""
    edges = list(graph.edges())
    draws = rng.random(2 * len(edges))
    adjacency = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        if draws[2 * i] < p:
            adjacency[u].append(v)
        if draws[2 * i + 1] < p:
            adjacency[v].append(u)
    return adjacency


def sample_live_edge_graph_lt(graph: nx.Graph, rng: np.random.Generator) -> LiveEdgeGraph:
    """Sample one live-edge graph under the Linear Threshold model.

    Each node with degree > 0 selects exactly one neighbor uniformly at
    random as its live in-arc source (uniform edge weights 1/degree(v)).
    """
    adjacency = defaultdict(list)
    for v in graph.nodes():
        neighbors = list(graph.neighbors(v))
        if not neighbors:
            continue
        chosen = neighbors[int(rng.integers(len(neighbors)))]
        adjacency[chosen].append(v)
    return adjacency


def sample_live_edge_graph(graph: nx.Graph, model: str, rng: np.random.Generator, p: float = 0.01) -> LiveEdgeGraph:
    model = model.upper()
    if model == "IC":
        return sample_live_edge_graph_ic(graph, p, rng)
    if model == "LT":
        return sample_live_edge_graph_lt(graph, rng)
    raise ValueError(f"Unknown diffusion model: {model!r} (expected 'IC' or 'LT')")


def simulate_ic(graph: nx.Graph, seeds, p: float = 0.01, rng: np.random.Generator | None = None) -> set:
    """Run a single Independent Cascade simulation, return the activated set."""
    rng = rng or np.random.default_rng()
    live_graph = sample_live_edge_graph_ic(graph, p, rng)
    return reachable_from(live_graph, seeds)


def simulate_lt(graph: nx.Graph, seeds, rng: np.random.Generator | None = None) -> set:
    """Run a single Linear Threshold simulation, return the activated set."""
    rng = rng or np.random.default_rng()
    live_graph = sample_live_edge_graph_lt(graph, rng)
    return reachable_from(live_graph, seeds)
