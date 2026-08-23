"""Monte Carlo expected-spread estimation and timing utilities.

All spread estimates in this project are Monte Carlo averages over a fixed
number of diffusion simulations (``mc_runs``), using a fixed random seed for
reproducibility. This module is the single place where that estimate is
computed, so every seed-selection method (greedy and all heuristics) is
scored with exactly the same evaluation procedure -- an apples-to-apples
comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import networkx as nx
import numpy as np

from src.diffusion_models import reachable_from, sample_live_edge_graph

DEFAULT_MC_RUNS = 200
DEFAULT_IC_PROBABILITY = 0.01


def generate_live_edge_graphs(
    graph: nx.Graph,
    model: str,
    num_samples: int,
    seed: int,
    p: float = DEFAULT_IC_PROBABILITY,
) -> list:
    """Pre-sample ``num_samples`` live-edge graphs (see diffusion_models.py).

    Reusing the same fixed set of sampled live-edge graphs across every
    marginal-gain evaluation within one greedy run (and across every
    heuristic's final spread evaluation) keeps the comparison apples-to-apples
    and is what makes Monte Carlo greedy tractable on this graph size.
    """
    rng = np.random.default_rng(seed)
    return [sample_live_edge_graph(graph, model, rng, p=p) for _ in range(num_samples)]


def estimate_spread_from_live_graphs(live_graphs: list, seeds) -> float:
    """Average reachable-set size from ``seeds`` across pre-sampled live-edge graphs."""
    seeds = list(seeds)
    if not seeds:
        return 0.0
    total = sum(len(reachable_from(lg, seeds)) for lg in live_graphs)
    return total / len(live_graphs)


def estimate_spread(
    graph: nx.Graph,
    seeds,
    model: str,
    mc_runs: int = DEFAULT_MC_RUNS,
    seed: int = 42,
    p: float = DEFAULT_IC_PROBABILITY,
) -> float:
    """Monte Carlo expected spread of a seed set, sampling fresh live-edge graphs.

    This is the function used for final, reported spread numbers for every
    method in results_table.csv, so all methods are scored identically
    regardless of how their seed set was chosen.
    """
    live_graphs = generate_live_edge_graphs(graph, model, mc_runs, seed, p=p)
    return estimate_spread_from_live_graphs(live_graphs, seeds)


@dataclass
class Timed:
    result: object
    seconds: float


def time_call(fn, *args, **kwargs) -> Timed:
    """Call ``fn`` and record wall-clock runtime in seconds."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return Timed(result=result, seconds=elapsed)
