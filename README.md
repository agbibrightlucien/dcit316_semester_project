# Budget-Constrained Influence Maximization

**How close can cheap centrality heuristics get to the greedy algorithm?**

DCIT 316 (Computational Models for Social Media Mining) semester project,
University of Ghana, Department of Computer Science. Student: Bright Lucien
Jnr Agbi.


## Research question

I wanted to answer a practical question. In a social network marketing or seeding scenario where I can only realistically target a small number of accounts, a budget, how does the influence spread from cheap, fast to compute centrality heuristics (degree, PageRank, betweenness, eigenvector centrality) compare to the spread from the classic greedy algorithm (Kempe, Kleinberg and Tardos, 2003), which is provably near optimal but computationally expensive. I also wanted to know how much compute time I save by using a heuristic instead.

## Dataset

[SNAP ego-Facebook](http://snap.stanford.edu/data/ego-Facebook.html)
(`facebook_combined.txt`): 4,039 nodes, 88,234 edges, a single connected
undirected graph. See [`data_card.md`](data_card.md) for full provenance,
stats, and limitations, and [`ethics_statement.md`](ethics_statement.md)
for the ethics/privacy statement.

## Methods compared

- **Greedy (KKT baseline):** the algorithm from Kempe, Kleinberg & Tardos
  (2003), implemented exactly (not approximated) but accelerated with the
  CELF lazy-forward optimization (Leskovec et al., 2007) so it is tractable
  on this graph size. See the module docstring in
  [`src/seed_selection.py`](src/seed_selection.py) for the full
  explanation of why this is a performance optimization and not a
  different algorithm.
- Degree centrality, PageRank, betweenness centrality (computed exactly), and eigenvector centrality. I took the top k nodes by each ranking.
- A random baseline, a fixed seed random k node sample, as a sanity check lower bound.

Both **Independent Cascade (IC)** and **Linear Threshold (LT)** diffusion
models are used (see [`src/diffusion_models.py`](src/diffusion_models.py)),
at budgets **k = 5, 10, 20, 50**.

## Implementation notes worth knowing before reading the code

1. **Live-edge graph equivalence.** I implemented both IC and LT simulations via their equivalent "live-edge graph" formulation from KKT (2003): one simulation = sample a random live-edge subgraph once, then compute BFS reachability from the seed set. This is mathematically identical to directly simulating the cascade, but lets live-edge graphs be reused across many marginal-gain evaluations during greedy selection.
2. **Greedy is run once per diffusion model, not once per budget.** I ran greedy once per diffusion model, not once per budget. Because greedy adds seeds one at a time in strictly decreasing marginal-gain order, its first *k* picks for budget *k*=50 are identical to what it would pick if run directly for any smaller budget. Centrality heuristics have the same property (a top-k slice of one ranking). So each ranking/greedy-order is computed once and sliced for every budget. This is a correctness preserving performance optimization, documented in `src/seed_selection.py`.
3. **All methods are scored identically.** I scored all methods identically. Regardless of how a method's seed set was chosen, every seed set's *reported* spread in `reports/results_table.csv` comes from the same function (`src/evaluation.py::estimate_spread`), using fresh Monte Carlo samples, an apples to apples comparison.
4. **Betweenness centrality is computed exactly.** I computed betweenness centrality exactly (not sampled). I timed it at ~86 seconds on this graph, which turned out to be affordable. See [`failure_log.md`](failure_log.md) #2.
5. **Fixed random seeds throughout.** I used fixed random seeds throughout (`RANDOM_SEED = 42` in `scripts/run_full_analysis.py`) for every stochastic step: Monte Carlo spread simulation, greedy's internal live-edge sampling, the random baseline, and betweenness's Brandes'-algorithm RNG.

## Repository structure

```
README.md                     # this file
data_card.md                  # dataset provenance, stats, limitations
ethics_statement.md           # ethics/privacy statement
failure_log.md                # issues hit during development and how they were resolved
conftest.py                   # pytest path setup
requirements.txt
data/
  facebook_combined.txt       # the dataset
notebooks/
  01_influence_maximization_analysis.ipynb   # exploratory notebook (executed, outputs saved)
src/
  graph_utils.py              # graph loading, descriptive stats
  diffusion_models.py         # IC and LT models (live-edge graph formulation)
  seed_selection.py           # greedy (CELF) + all heuristics + random baseline
  evaluation.py                # Monte Carlo spread estimation, timing
scripts/
  run_full_analysis.py        # end-to-end pipeline: run this to reproduce everything
reports/
  graph_stats.json            # basic graph statistics
  results_table.csv           # every method x budget x diffusion-model combination
  interpretation.md           # my written summary of the results
  figures/
    spread_vs_budget.png
    runtime_vs_budget.png
    efficiency_chart.png
tests/
  test_functions.py           # unit tests (toy graphs)
  test_submission.py          # integration tests (real dataset, small/fast parameters)
```

## How to reproduce

```bash
pip install -r requirements.txt

# Regenerate reports/results_table.csv, reports/figures/*, reports/interpretation.md
python3 scripts/run_full_analysis.py     # ~9 minutes on this graph

# Run the test suite
python3 -m pytest tests/ -q

# Re-execute the notebook end to end
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/01_influence_maximization_analysis.ipynb
```

The dataset is already included at `data/facebook_combined.txt`. To fetch it
from scratch instead: `git clone https://github.com/jcatw/snap-facebook` and
copy `facebook_combined.txt` into `data/` (see `data_card.md`).

## Summary of findings

Full numbers are in [`reports/results_table.csv`](reports/results_table.csv) and my written summary is in [`reports/interpretation.md`](reports/interpretation.md), including a section on a few surprising findings that I chose not to hide. Headline points:

- **Degree centrality was the standout cheap heuristic for me.** Across both diffusion models and all four budgets, it reaches roughly **84–102%** of greedy's spread while costing on the order of **0.01–0.03%** of greedy's runtime, multiple orders of magnitude cheaper for spread that is usually within about 10-20% of near-optimal.
- **PageRank is close behind degree** (roughly 83–100% of greedy's spread) at a still-tiny runtime cost (~0.9–1.3% of greedy's).
- **Betweenness centrality is not worth its cost here.** It reaches similar spread quality to degree/PageRank (~82–92%) but its exact computation costs **more wall-clock time than greedy itself** (about 5–8x greedy's runtime), the runtime-vs-quality tradeoff clearly favors degree or PageRank over betweenness on this graph.
- **Eigenvector centrality underperforms** relative to the other heuristics (as low as ~21–68% of greedy's spread), plausibly because it concentrates seeds within one densely-connected cluster of the ego network rather than spreading across the graph's structure.
- **The random baseline** is a genuine lower bound, well below every informed method, and, as expected, closes the gap somewhat as the budget grows (more random seeds means a higher chance of hitting a well-connected node by chance).
- **Runtime scales very differently across methods.** Greedy's runtime grows with the budget k (more seeds selected = more CELF iterations); every heuristic's runtime is dominated by a *one-time* ranking computation and is essentially flat across budgets. A business operating under a compute-time constraint, not just a seed-count constraint, gets that benefit at any budget size, not only small ones.
- **A few (method, budget) combinations had a heuristic match or slightly exceed greedy's reported spread** (see `reports/interpretation.md` for the exact list). This is flagged rather than hidden, see [`failure_log.md`](failure_log.md) #4 for the likely explanation (Monte Carlo sampling variance at small absolute spread values) and a suggested framing for the paper's limitations section.

I used these results, figures, and my interpretation notes as the basis for writing the actual paper separately.
