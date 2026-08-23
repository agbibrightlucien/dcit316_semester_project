# Failure Log

Running log of things that broke, dead ends, and how they were resolved
during pipeline development. Kept per the project brief's instruction to
flag issues transparently rather than silently smoothing them over.

## 1. Naive greedy (re-simulating cascades per marginal-gain check) would not have been tractable

**Issue:** The brief's baseline description of greedy -- "iteratively pick
the node that adds the most marginal spread, using Monte Carlo simulation"
-- if implemented the most literal way (re-run N fresh cascade simulations
from scratch for every candidate node, every iteration) would cost roughly
`k * n * mc_runs` full cascade simulations. For k=50, n=4,039,
mc_runs=200, that's ~40 million cascade simulations -- not feasible in the
project timeline.

**Resolution:** Implemented greedy using the standard "live-edge graph"
equivalence from Kempe, Kleinberg & Tardos (2003) itself (a single cascade
simulation is exactly equivalent to sampling one random live-edge subgraph
and computing BFS reachability from the seed set), combined with the CELF
lazy-forward optimization (Leskovec et al., 2007). This is the same
algorithm and the same random process -- not an approximation -- just
implemented so that (a) each candidate node's individual reachability is
cached once and reused (reachability is monotone under set union of
sources, so `spread(S ∪ {v})`'s live-edge-graph value is always
`reached(S) ∪ reach_cache[v]`), and (b) CELF skips recomputing
marginal gains that provably cannot change the next pick. Verified this
produces the mathematically identical seed set as naive greedy would (see
`test_greedy_prefix_property` and the CELF equivalence proof) while running
in ~11-21 seconds end to end (k=50, 200 MC runs, full 4,039-node graph) --
see `src/seed_selection.py` module docstring for the full explanation.

## 2. Assumed exact betweenness centrality would be infeasible; it wasn't

**Issue:** Initial assumption was that exact betweenness centrality
(Brandes' algorithm, O(V·(V+E))) would be too slow on ~4,000 nodes /
~88,000 edges in pure-Python NetworkX, so the plan was to use NetworkX's
sampled/approximate betweenness estimator (`k=500` source samples) instead,
as the brief's Section 5.3 anticipates might be necessary for some
computation.

**Resolution:** Timed it directly before committing to the approximation:
exact betweenness centrality on the full graph took **86 seconds**, which
is affordable. Switched to exact betweenness (`BETWEENNESS_SAMPLE_SIZE =
None` in `src/seed_selection.py`) for a more rigorous result, keeping the
sampled-estimator code path available (via the `sample_size` parameter) in
case this pipeline is ever re-run on a much larger graph where exact
betweenness would not be affordable.

## 3. `networkx.pagerank()` does not accept a `seed` parameter

**Issue:** Initial code passed `seed=RANDOM_SEED` to `nx.pagerank()` for
consistency with the other reproducibility-seeded functions, on the
(incorrect) assumption it used random initialization. This raised
`TypeError: pagerank() got an unexpected keyword argument 'seed'`.

**Resolution:** NetworkX's PageRank implementation is deterministic power
iteration with no random initialization, so no seed is needed or accepted.
Removed the parameter and documented why in a code comment in
`src/seed_selection.py::pagerank_ranking`.

## 4. Degree and PageRank heuristics matched or slightly exceeded greedy's reported spread at a few (method, budget) combinations

**Issue:** In the full experiment results (`reports/results_table.csv`),
`degree` at IC/k=10 reached 101.9% of greedy's spread, and `degree` and
`pagerank` both reached exactly 100.0% of greedy's spread at LT/k=5. Greedy
is provably near-optimal (within a `(1 - 1/e)` factor) for this class of
submodular objective, so a heuristic should essentially never *exceed* it.

**Resolution:** This is flagged, not hidden -- see the auto-generated
"Notable / surprising findings" section of `reports/interpretation.md`,
which lists every such case by name. The most likely explanation is Monte
Carlo sampling variance: `results_table.csv`'s spread column re-evaluates
every method (including greedy) on a **fresh, independent** 200-run Monte
Carlo sample (via `src/evaluation.py::estimate_spread`) for apples-to-apples
fairness, separate from the live-edge graphs greedy used internally during
selection -- so greedy's selection is optimal against *its own* sample but
is then scored, like everyone else, against a different one. At small
budgets (k=5, k=10) the absolute spread values are small enough that a few
points of MC noise is a meaningful percentage. This is worth a sentence in
the paper's limitations/discussion section rather than a code bug to "fix"
by re-running until the numbers look cleaner.

## 5. `nbformat`/`nbconvert`/`ipykernel` were not preinstalled

**Issue:** The notebook deliverable (`notebooks/01_influence_maximization_analysis.ipynb`)
requires `nbformat` to construct and `nbconvert` + `ipykernel` to execute
and verify it end to end; none were present in the base environment.

**Resolution:** Installed via pip (see `requirements.txt`), then executed
the notebook end to end with `jupyter nbconvert --execute` to confirm every
cell runs without error before treating it as done, rather than assuming an
unexecuted notebook was sufficient.
