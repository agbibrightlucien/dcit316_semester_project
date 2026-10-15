"""End-to-end pipeline: load graph -> select seeds -> evaluate -> report.

Run with:  python3 scripts/run_full_analysis.py

Produces:
  reports/graph_stats.json
  reports/results_table.csv
  reports/figures/spread_vs_budget.png
  reports/figures/runtime_vs_budget.png
  reports/figures/efficiency_chart.png
  reports/interpretation.md

Reproducibility: every stochastic step (Monte Carlo spread simulation,
greedy's internal live-edge sampling, the random baseline, betweenness's
source sampling) is seeded from RANDOM_SEED below.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import DEFAULT_IC_PROBABILITY, estimate_spread
from src.graph_utils import compute_basic_stats, load_graph, print_stats, save_stats
from src.seed_selection import (
    betweenness_ranking,
    degree_ranking,
    eigenvector_ranking,
    greedy_seeds_incremental,
    pagerank_ranking,
    random_seeds,
    ranking_to_seed_sets,
)

RANDOM_SEED = 42
BUDGETS = [5, 10, 20, 50]
MAX_BUDGET = max(BUDGETS)
MODELS = ["IC", "LT"]
MC_RUNS = 200  # Monte Carlo simulations per spread estimate (fixed for every method)
IC_EDGE_PROBABILITY = DEFAULT_IC_PROBABILITY  # 0.01, standard KKT parameterization

DATA_PATH = ROOT / "data" / "facebook_combined.txt"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Colorblind-safe categorical palette (fixed order, one color per method).
METHOD_COLORS = {
    "greedy": "#2a78d6",       # blue
    "degree": "#eb6834",       # orange
    "pagerank": "#1baf7a",     # aqua
    "betweenness": "#eda100",  # yellow
    "eigenvector": "#e87ba4",  # magenta
    "random": "#008300",       # green
}
METHOD_ORDER = ["greedy", "degree", "pagerank", "betweenness", "eigenvector", "random"]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def build_heuristic_seed_sets(graph):
    """Compute each heuristic's ranking once; time it once; slice for every budget."""
    heuristics = {}

    t0 = time.perf_counter()
    ranking = degree_ranking(graph)
    heuristics["degree"] = {"seed_sets": ranking_to_seed_sets(ranking, BUDGETS), "seconds": time.perf_counter() - t0}

    t0 = time.perf_counter()
    ranking = pagerank_ranking(graph)
    heuristics["pagerank"] = {"seed_sets": ranking_to_seed_sets(ranking, BUDGETS), "seconds": time.perf_counter() - t0}

    t0 = time.perf_counter()
    ranking = betweenness_ranking(graph, seed=RANDOM_SEED)
    heuristics["betweenness"] = {
        "seed_sets": ranking_to_seed_sets(ranking, BUDGETS),
        "seconds": time.perf_counter() - t0,
    }

    t0 = time.perf_counter()
    ranking = eigenvector_ranking(graph)
    heuristics["eigenvector"] = {
        "seed_sets": ranking_to_seed_sets(ranking, BUDGETS),
        "seconds": time.perf_counter() - t0,
    }

    t0 = time.perf_counter()
    ranking = random_seeds(graph, max_k=MAX_BUDGET, seed=RANDOM_SEED)
    heuristics["random"] = {"seed_sets": ranking_to_seed_sets(ranking, BUDGETS), "seconds": time.perf_counter() - t0}

    return heuristics


def run_experiment(graph) -> pd.DataFrame:
    rows = []

    for model in MODELS:
        print(f"\n=== Diffusion model: {model} ===")

        print(f"Running CELF greedy up to k={MAX_BUDGET} ({MC_RUNS} MC runs)...")
        greedy_result = greedy_seeds_incremental(
            graph, model=model, max_k=MAX_BUDGET, mc_runs=MC_RUNS, seed=RANDOM_SEED, p=IC_EDGE_PROBABILITY
        )
        print(f"  greedy total selection time: {greedy_result['cumulative_seconds'][-1]:.2f}s")

        print("Computing heuristic rankings...")
        heuristics = build_heuristic_seed_sets(graph)
        for name, info in heuristics.items():
            print(f"  {name}: ranking computed in {info['seconds']:.2f}s")

        for budget in BUDGETS:
            greedy_seed_set = greedy_result["order"][:budget]
            greedy_runtime = greedy_result["cumulative_seconds"][budget - 1]
            greedy_spread = estimate_spread(
                graph, greedy_seed_set, model=model, mc_runs=MC_RUNS, seed=RANDOM_SEED + 1, p=IC_EDGE_PROBABILITY
            )
            rows.append(
                {
                    "diffusion_model": model,
                    "method": "greedy",
                    "budget_k": budget,
                    "expected_spread": greedy_spread,
                    "runtime_seconds": greedy_runtime,
                    "seed_set": greedy_seed_set,
                }
            )

            for name, info in heuristics.items():
                seed_set = info["seed_sets"][budget]
                spread = estimate_spread(
                    graph, seed_set, model=model, mc_runs=MC_RUNS, seed=RANDOM_SEED + 1, p=IC_EDGE_PROBABILITY
                )
                rows.append(
                    {
                        "diffusion_model": model,
                        "method": name,
                        "budget_k": budget,
                        "expected_spread": spread,
                        "runtime_seconds": info["seconds"],
                        "seed_set": seed_set,
                    }
                )

    df = pd.DataFrame(rows)

    # Derived columns: how close to greedy, and how much cheaper than greedy.
    greedy_lookup = df[df.method == "greedy"].set_index(["diffusion_model", "budget_k"])
    df["spread_pct_of_greedy"] = df.apply(
        lambda r: 100.0 * r.expected_spread / greedy_lookup.loc[(r.diffusion_model, r.budget_k), "expected_spread"],
        axis=1,
    )
    df["runtime_pct_of_greedy"] = df.apply(
        lambda r: 100.0 * r.runtime_seconds / greedy_lookup.loc[(r.diffusion_model, r.budget_k), "runtime_seconds"],
        axis=1,
    )
    return df


def make_plots(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # (a) spread vs budget, one panel per diffusion model
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, model in zip(axes, MODELS):
        sub = df[df.diffusion_model == model]
        for method in METHOD_ORDER:
            m = sub[sub.method == method].sort_values("budget_k")
            ax.plot(
                m.budget_k, m.expected_spread, marker="o", markersize=5, linewidth=2,
                color=METHOD_COLORS[method], label=method,
            )
        ax.set_title(f"{model} model", color=INK, fontsize=11)
        ax.set_xlabel("Budget (k seeds)", color=INK)
        ax.set_xticks(BUDGETS)
        style_axes(ax)
    axes[0].set_ylabel("Expected spread (nodes)", color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Expected influence spread vs. seeding budget", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "spread_vs_budget.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # (b) runtime vs budget (log scale), one panel per diffusion model
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, model in zip(axes, MODELS):
        sub = df[df.diffusion_model == model]
        for method in METHOD_ORDER:
            m = sub[sub.method == method].sort_values("budget_k")
            ax.plot(
                m.budget_k, m.runtime_seconds, marker="o", markersize=5, linewidth=2,
                color=METHOD_COLORS[method], label=method,
            )
        ax.set_yscale("log")
        ax.set_title(f"{model} model", color=INK, fontsize=11)
        ax.set_xlabel("Budget (k seeds)", color=INK)
        ax.set_xticks(BUDGETS)
        style_axes(ax)
    axes[0].set_ylabel("Wall-clock runtime (seconds, log scale)", color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Seed-set computation runtime vs. seeding budget", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "runtime_vs_budget.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # (c) efficiency: spread achieved per second of compute
    df = df.copy()
    df["spread_per_second"] = df.expected_spread / df.runtime_seconds
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, model in zip(axes, MODELS):
        sub = df[df.diffusion_model == model]
        for method in METHOD_ORDER:
            m = sub[sub.method == method].sort_values("budget_k")
            ax.plot(
                m.budget_k, m.spread_per_second, marker="o", markersize=5, linewidth=2,
                color=METHOD_COLORS[method], label=method,
            )
        ax.set_yscale("log")
        ax.set_title(f"{model} model", color=INK, fontsize=11)
        ax.set_xlabel("Budget (k seeds)", color=INK)
        ax.set_xticks(BUDGETS)
        style_axes(ax)
    axes[0].set_ylabel("Spread per second of compute (log scale)", color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Compute efficiency: influence spread per second", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "efficiency_chart.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_interpretation(df: pd.DataFrame, graph_stats: dict) -> None:
    lines = []
    lines.append("# Interpretation of results\n")
    lines.append(
        "This is my written summary of the results. The numbers below are read directly "
        "from results_table.csv, not estimated by hand. This is a technical summary "
        "I used while writing the paper, not the paper text itself.\n"
    )

    lines.append("## Dataset\n")
    lines.append(
        f"SNAP ego-Facebook: {graph_stats['num_nodes']} nodes, {graph_stats['num_edges']} edges, "
        f"average degree {graph_stats['avg_degree']:.2f}, density {graph_stats['density']:.5f}, "
        f"{graph_stats['num_connected_components']} connected component(s) "
        f"(largest holds {graph_stats['largest_component_fraction']*100:.1f}% of nodes), "
        f"average clustering coefficient {graph_stats['average_clustering_coefficient']:.4f}.\n"
    )

    lines.append("## Key numbers: how close do cheap heuristics get to greedy?\n")
    for model in MODELS:
        lines.append(f"### {model} model\n")
        sub = df[(df.diffusion_model == model) & (df.method != "greedy")]
        lines.append("| Method | Budget k | Spread (% of greedy) | Runtime (% of greedy) |")
        lines.append("|---|---|---|---|")
        for _, r in sub.sort_values(["method", "budget_k"]).iterrows():
            lines.append(
                f"| {r.method} | {r.budget_k} | {r.spread_pct_of_greedy:.1f}% | {r.runtime_pct_of_greedy:.3f}% |"
            )
        lines.append("")

    lines.append("## Notable / surprising findings\n")
    flagged = df[(df.method != "greedy") & (df.spread_pct_of_greedy >= 100.0)]
    if len(flagged):
        lines.append(
            "The following (method, model, budget) combinations matched or "
            "**exceeded** greedy's Monte Carlo spread estimate. This can genuinely "
            "happen because both numbers are Monte Carlo estimates with sampling "
            "variance (see `MC_RUNS` in `scripts/run_full_analysis.py`), or because "
            "greedy is optimizing on its own internal live-edge sample while the "
            "table's spread column re-evaluates every method on a fresh, "
            "independent MC sample for fairness, it is not silently smoothed "
            "over here and is worth a sentence in the paper's discussion/limitations:\n"
        )
        for _, r in flagged.iterrows():
            lines.append(
                f"- {r.method} / {r.diffusion_model} / k={r.budget_k}: "
                f"{r.spread_pct_of_greedy:.1f}% of greedy's spread"
            )
    else:
        lines.append("No heuristic matched or exceeded greedy's spread at any (model, budget) combination.")
    lines.append("")

    lines.append("## Suggested connections to cited literature (for the paper, not filled in here)\n")
    lines.append(
        "- The efficiency-vs-quality tradeoff shown in `efficiency_chart.png` is exactly the "
        "motivation for scalable/heuristic influence maximization follow-up work after "
        "Kempe, Kleinberg & Tardos (2003); this is the place in the paper to cite such follow-ups.\n"
        "- The CELF (lazy-forward) optimization used to make greedy tractable here "
        "(see `src/seed_selection.py` module docstring) is from Leskovec et al. (2007), "
        "'Cost-effective Outbreak Detection in Networks', cite it specifically for the "
        "greedy implementation, not just KKT (2003).\n"
        "- Runtime for centrality heuristics is dominated by a one-time ranking computation "
        "and barely changes with budget k, while greedy's runtime scales with k. This "
        "contrast is worth a sentence discussing why heuristics are attractive specifically "
        "under tight compute budgets, independent of how small k is.\n"
    )

    (REPORTS_DIR / "interpretation.md").write_text("\n".join(lines))


def main():
    print("Loading graph...")
    graph = load_graph(DATA_PATH)
    stats = compute_basic_stats(graph)
    print_stats(stats)
    save_stats(stats, REPORTS_DIR / "graph_stats.json")

    df = run_experiment(graph)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_df = df.drop(columns=["seed_set"])
    csv_df.to_csv(REPORTS_DIR / "results_table.csv", index=False)
    print(f"\nSaved results table: {REPORTS_DIR / 'results_table.csv'}")

    make_plots(df)
    print(f"Saved figures to: {FIGURES_DIR}")

    write_interpretation(df, stats)
    print(f"Saved interpretation: {REPORTS_DIR / 'interpretation.md'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
