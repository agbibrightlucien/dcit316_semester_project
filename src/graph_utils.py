"""Graph loading and basic descriptive statistics for the ego-Facebook dataset.

Dataset: SNAP ego-Facebook (facebook_combined.txt), McAuley & Leskovec (2012),
Leskovec & Krevl (2014). Undirected, unweighted edge list, one edge per line
("u v" = an edge between node u and node v).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import networkx as nx


def load_graph(edge_list_path: str | Path) -> nx.Graph:
    """Load an undirected edge-list file into a NetworkX Graph.

    File format: whitespace-separated "u v" per line, integer node IDs.
    """
    edge_list_path = Path(edge_list_path)
    if not edge_list_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {edge_list_path}. Expected facebook_combined.txt "
            "(see data_card.md for the source)."
        )
    graph = nx.read_edgelist(edge_list_path, nodetype=int, create_using=nx.Graph())
    return graph


def compute_basic_stats(graph: nx.Graph) -> dict:
    """Compute standard descriptive statistics for a graph.

    Clustering coefficient and connected-component counts are relatively
    expensive on graphs of this size, so wall-clock time for computing this
    stats block is also recorded for transparency.
    """
    start = time.time()

    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    degrees = [d for _, d in graph.degree()]
    avg_degree = sum(degrees) / num_nodes if num_nodes else 0.0
    density = nx.density(graph)
    components = list(nx.connected_components(graph))
    num_components = len(components)
    largest_component_size = max((len(c) for c in components), default=0)
    avg_clustering = nx.average_clustering(graph)

    elapsed = time.time() - start

    stats = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "min_degree": min(degrees) if degrees else 0,
        "max_degree": max(degrees) if degrees else 0,
        "density": density,
        "num_connected_components": num_components,
        "largest_component_size": largest_component_size,
        "largest_component_fraction": (
            largest_component_size / num_nodes if num_nodes else 0.0
        ),
        "average_clustering_coefficient": avg_clustering,
        "stats_computation_seconds": elapsed,
    }
    return stats


def save_stats(stats: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)


def print_stats(stats: dict) -> None:
    print("Graph basic statistics")
    print("-----------------------")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
