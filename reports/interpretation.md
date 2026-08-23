# Interpretation of results

Auto-generated from the actual pipeline output (`scripts/run_full_analysis.py`) -- numbers below are read directly from `results_table.csv`, not estimated by hand. This is a technical summary for the paper draft, not the paper text itself.

## Dataset

SNAP ego-Facebook: 4039 nodes, 88234 edges, average degree 43.69, density 0.01082, 1 connected component(s) (largest holds 100.0% of nodes), average clustering coefficient 0.6055.

## Key numbers: how close do cheap heuristics get to greedy?

### IC model

| Method | Budget k | Spread (% of greedy) | Runtime (% of greedy) |
|---|---|---|---|
| betweenness | 5 | 88.0% | 499.908% |
| betweenness | 10 | 81.6% | 496.325% |
| betweenness | 20 | 83.1% | 492.223% |
| betweenness | 50 | 89.2% | 486.319% |
| degree | 5 | 89.5% | 0.014% |
| degree | 10 | 101.9% | 0.014% |
| degree | 20 | 98.3% | 0.014% |
| degree | 50 | 86.3% | 0.014% |
| eigenvector | 5 | 68.1% | 4.222% |
| eigenvector | 10 | 61.0% | 4.192% |
| eigenvector | 20 | 55.5% | 4.157% |
| eigenvector | 50 | 47.8% | 4.107% |
| pagerank | 5 | 89.5% | 0.954% |
| pagerank | 10 | 83.2% | 0.947% |
| pagerank | 20 | 83.1% | 0.939% |
| pagerank | 50 | 93.5% | 0.928% |
| random | 5 | 15.6% | 0.002% |
| random | 10 | 39.9% | 0.002% |
| random | 20 | 73.0% | 0.002% |
| random | 50 | 79.9% | 0.002% |

### LT model

| Method | Budget k | Spread (% of greedy) | Runtime (% of greedy) |
|---|---|---|---|
| betweenness | 5 | 88.2% | 831.411% |
| betweenness | 10 | 88.1% | 826.677% |
| betweenness | 20 | 91.5% | 813.644% |
| betweenness | 50 | 86.3% | 781.454% |
| degree | 5 | 100.0% | 0.025% |
| degree | 10 | 93.0% | 0.025% |
| degree | 20 | 96.1% | 0.024% |
| degree | 50 | 83.9% | 0.023% |
| eigenvector | 5 | 24.1% | 5.961% |
| eigenvector | 10 | 24.5% | 5.927% |
| eigenvector | 20 | 23.9% | 5.833% |
| eigenvector | 50 | 20.8% | 5.603% |
| pagerank | 5 | 100.0% | 1.348% |
| pagerank | 10 | 99.1% | 1.340% |
| pagerank | 20 | 96.2% | 1.319% |
| pagerank | 50 | 94.9% | 1.267% |
| random | 5 | 6.3% | 0.003% |
| random | 10 | 8.8% | 0.003% |
| random | 20 | 26.4% | 0.003% |
| random | 50 | 35.6% | 0.003% |

## Notable / surprising findings

The following (method, model, budget) combinations matched or **exceeded** greedy's Monte Carlo spread estimate. This can genuinely happen because both numbers are Monte Carlo estimates with sampling variance (see `MC_RUNS` in `scripts/run_full_analysis.py`), or because greedy is optimizing on its own internal live-edge sample while the table's spread column re-evaluates every method on a fresh, independent MC sample for fairness -- it is not silently smoothed over here and is worth a sentence in the paper's discussion/limitations:

- degree / IC / k=10: 101.9% of greedy's spread
- degree / LT / k=5: 100.0% of greedy's spread
- pagerank / LT / k=5: 100.0% of greedy's spread

## Suggested connections to cited literature (for the paper, not filled in here)

- The efficiency-vs-quality tradeoff shown in `efficiency_chart.png` is exactly the motivation for scalable/heuristic influence maximization follow-up work after Kempe, Kleinberg & Tardos (2003); this is the place in the paper to cite such follow-ups.
- The CELF (lazy-forward) optimization used to make greedy tractable here (see `src/seed_selection.py` module docstring) is from Leskovec et al. (2007), 'Cost-effective Outbreak Detection in Networks' -- cite it specifically for the greedy implementation, not just KKT (2003).
- Runtime for centrality heuristics is dominated by a one-time ranking computation and barely changes with budget k, while greedy's runtime scales with k -- this contrast is worth a sentence discussing why heuristics are attractive specifically under tight compute budgets, independent of how small k is.
