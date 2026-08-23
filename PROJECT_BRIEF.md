# Project Brief: Budget-Constrained Influence Maximization

## Purpose of this document
This is a handoff brief for Claude Code. It explains the full context, decisions already made, and what needs to be built. Read this fully before writing any code.

---

## 1. Academic context

This is a research paper project for **DCIT 316 (Computational Models for Social Media Mining)**, a Level 300 university course at the University of Ghana, Department of Computer Science. The student is Bright Lucien Jnr Agbi.

Timeline: the paper needs to be built and written within a few days, so the pipeline needs to work correctly on the first pass, not require extensive debugging cycles.

The final output of the overall project is a research paper. Claude Code's job is to build the technical pipeline and produce the results, tables, and figures that the paper will be written around. Claude Code is not expected to write the paper itself.

## 2. How this topic was chosen (important context, do not deviate)

The student needed a topic that satisfies three hard requirements:
1. Uses a **real, publicly and immediately accessible dataset** (no gated access, no permission requests, no multi-day approval waits).
2. Is either a **genuine extension of existing published work** or a **genuinely novel angle**, not a repeat of something already done.
3. Can **cite real, verifiable academic papers** in the specific domain.

Two earlier topic directions were explicitly rejected and should NOT be revisited:
- **Any topic involving the Twi language or Ghanaian code-switched text** (e.g. sentiment analysis on AsanteTwiSenti). This was ruled out because the student cannot speak or validate Twi themselves, and a prior related topic (hate speech in code-switched Ghanaian text) was rejected by the course lecturer for this exact reason.
- **Twitter bot detection using Cresci-2017 or TwiBot-20.** This was explored and abandoned because: (a) TwiBot-20's full dataset requires emailing the author for permission, which is not reliable within the project timeline, and (b) Cresci-2017 has been shown in published literature (Rauchfleisch & Kaiser-style dataset critique papers) to be trivially separable by a single yes/no decision rule, meaning there's no real accuracy gap left to study, undermining the intended research question.

The chosen topic (below) was selected specifically because it avoids both problems: the dataset is verified to be immediately downloadable with no gate, and the research question has real, unsaturated room for a genuine contribution.

## 3. The chosen topic

**Working title:** *Budget-Constrained Influence Maximization: How Close Can Cheap Centrality Heuristics Get to the Greedy Algorithm?*

### Research question
In social network marketing/seeding scenarios where only a small number of accounts can realistically be targeted (a "budget"), how does the influence spread achieved by cheap, fast-to-compute centrality heuristics (degree centrality, PageRank, betweenness centrality, eigenvector centrality) compare to the spread achieved by the classic greedy algorithm (Kempe, Kleinberg & Tardos, 2003), which is provably near-optimal but computationally expensive?

### Business framing
If a business or campaign has a limited budget to seed a marketing push (e.g. can only pay or approach 5, 10, or 20 accounts), which cheap heuristic gets them closest to the influence spread of the expensive, near-optimal greedy solution, and how much compute time do they save by not running greedy?

### Why this is a genuine contribution, not a repeat
This is not proposing a new algorithm. It is a rigorous, quantified cost-vs-benefit benchmark of existing heuristics against the greedy baseline, framed explicitly around a realistic budget constraint, with real runtime measurements alongside spread quality. The novelty is in the systematic efficiency/quality tradeoff analysis under this specific framing, not in inventing a new method.

### Foundational and citable literature
- Kempe, D., Kleinberg, J., & Tardos, É. (2003). "Maximizing the spread of influence through a social network." Proceedings of the 9th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 137-146. (This is the foundational paper that introduced the greedy algorithm and the Independent Cascade / Linear Threshold diffusion models. Cite this for the core methodology.)
- Leskovec, J., & Krevl, A. (2014). "SNAP Datasets: Stanford Large Network Dataset Collection." http://snap.stanford.edu/data. (Cite this for the dataset source.)
- McAuley, J., & Leskovec, J. (2012). "Learning to Discover Social Circles in Ego Networks." NIPS 2012. (This is the original paper describing the Facebook ego-network dataset itself, cite this alongside the SNAP citation.)
- Any efficiency-focused influence maximization follow-up work (e.g. work on scalable/parallel influence maximization, or heuristic-vs-greedy comparison papers) can be searched for and cited to support the "this tradeoff is an active area of study" framing. Claude Code does not need to find these citations, that will be handled separately when the paper is written, but should leave clear notes on which specific results in the output would benefit from being connected to such literature.

## 4. Dataset

**Dataset:** SNAP ego-Facebook (also called facebook_combined.txt)
**Stats:** 4,039 nodes, 88,234 edges, undirected graph.
**Original source:** http://snap.stanford.edu/data/ego-Facebook.html (not directly reachable from a sandboxed environment with restricted network access, but the data itself is public and freely licensed for research use).
**Verified working mirror:** https://github.com/jcatw/snap-facebook — this repository has the actual `facebook_combined.txt` file committed directly (not gitignored), confirmed working via `git clone`.

**Instructions for Claude Code:**
1. Clone or download `facebook_combined.txt` from `https://github.com/jcatw/snap-facebook`.
2. File format: plain text edge list, one edge per line, space-separated node IDs, e.g. `0 1` means node 0 is connected to node 1. Undirected.
3. Treat this as the primary dataset. Do not attempt to use Cresci-2017, TwiBot-20, or any Twi-language dataset (see Section 2 for why).
4. If a second, larger network dataset is wanted later for a robustness check, other SNAP ego-networks (e.g. ego-Twitter, which is directed) exist in the same style of GitHub mirrors, but this is optional and not required for the core deliverable.

## 5. Technical requirements

### 5.1 Core pipeline components
1. **Graph loading and basic stats** — load the edge list into a graph structure (e.g. NetworkX), report basic descriptive stats (node count, edge count, average degree, density, connected components, clustering coefficient).
2. **Diffusion models** — implement both:
   - Independent Cascade (IC) model
   - Linear Threshold (LT) model
   Both are standard, well-documented models from the Kempe-Kleinberg-Tardos paper. Use standard parameterizations (e.g. uniform edge activation probability for IC, e.g. p = 0.01 or p = 1/in-degree; uniform random thresholds for LT).
3. **Seed selection methods to compare:**
   - Greedy algorithm (the KKT baseline): iteratively pick the node that adds the most marginal spread, using Monte Carlo simulation of the diffusion process to estimate expected spread. This is expensive, that expense is the whole point of the comparison, so implement it correctly even though it's slow.
   - Degree centrality (top-k highest degree nodes)
   - PageRank (top-k highest PageRank nodes)
   - Betweenness centrality (top-k highest betweenness nodes)
   - Eigenvector centrality (top-k highest eigenvector centrality nodes)
   - A random baseline (random k nodes) as a sanity-check lower bound
4. **Budget levels to test:** run all seed selection methods at multiple budget sizes, e.g. k = 5, 10, 20, 50. This lets the paper show how the gap between heuristics and greedy changes as budget grows.
5. **Evaluation metrics for each method at each budget level:**
   - Expected influence spread (via Monte Carlo simulation, e.g. average over 100–1000 simulation runs, report this parameter clearly)
   - Wall-clock runtime to compute the seed set
   - Spread as a percentage of the greedy result (this is the key "how close do we get" number)
   - Runtime as a percentage/multiple of the greedy runtime (this is the key "how much cheaper" number)

### 5.2 Output requirements
- A clean, well-commented, reproducible codebase (Python, using NetworkX and standard scientific libraries: numpy, pandas, matplotlib).
- A results table (CSV or similar) with all method × budget-level combinations and their metrics.
- Plots: at minimum, (a) spread vs. budget size for each method on one chart, (b) runtime vs. budget size for each method on one chart (likely log-scale for runtime), (c) a "spread achieved per second of compute" efficiency chart if feasible.
- All randomness (Monte Carlo simulations, random baseline) must use a fixed random seed for reproducibility, and this should be clearly stated in the code and output.
- Code should be organized into clear modules or clearly separated notebook sections: data loading, diffusion models, seed selection methods, evaluation/simulation, results generation, plotting.

### 5.3 What NOT to do
- Do not use any Twi-language, AsanteTwiSenti, or AfriSenti data or methods.
- Do not use Cresci-2017, TwiBot-20, or any Twitter bot detection dataset.
- Do not invent a new algorithm or claim state-of-the-art performance. The contribution is the benchmark and analysis, not a new method.
- Do not skip the greedy algorithm because it's slow. If full greedy on the full 4,039-node graph is too slow to run at all budget levels, that's fine, reduce the number of Monte Carlo simulation runs for greedy specifically, or run greedy on a smaller random subgraph sample as a documented limitation, but attempt the full pipeline first before deciding it's infeasible.

## 6. Repository structure

This project should follow the same structural convention the student has used in prior DCIT 316 individual assignments (e.g. the IA4 diffusion/homophily/recommendation repo), so it is consistent with their existing GitHub organization and workflow. Match this layout as closely as possible:

```
/
├── PROJECT_BRIEF.md              # this document (equivalent to ASSIGNMENT_BRIEF.md in prior IAs)
├── README.md                     # project overview, how to run, summary of findings once complete
├── data_card.md                  # documents the dataset: source, license, stats, collection method,
│                                  #   known limitations (see Section 4 of this brief for the content)
├── ethics_statement.md           # brief statement on ethical use of the public SNAP dataset
│                                  #   (anonymized node IDs, no personal data, research-use license)
├── failure_log.md                # running log of things that broke, dead ends, and how they were
│                                  #   resolved (e.g. if greedy is too slow on full graph, log it here)
├── conftest.py                   # pytest configuration/fixtures
├── requirements.txt
├── data/
│   └── facebook_combined.txt     # the dataset (or a subfolder if versioned, matching prior convention)
├── notebooks/
│   └── 01_influence_maximization_analysis.ipynb   # main exploratory/analysis notebook
├── src/
│   ├── graph_utils.py            # loading, basic stats
│   ├── diffusion_models.py       # IC and LT implementations
│   ├── seed_selection.py         # greedy + all heuristic methods
│   └── evaluation.py             # Monte Carlo spread estimation, timing
├── scripts/
│   └── run_full_analysis.py      # main script that runs everything end to end, mirrors prior IA pattern
├── reports/
│   ├── results_table.csv
│   ├── interpretation.md         # written interpretation of results, matches prior IA pattern
│   └── figures/
│       ├── spread_vs_budget.png
│       ├── runtime_vs_budget.png
│       └── efficiency_chart.png
└── tests/
    ├── test_functions.py         # unit tests for diffusion models, seed selection, evaluation logic
    └── test_submission.py        # end-to-end/integration-style checks, matches prior IA pattern
```

## 7. Deliverable checklist for Claude Code

- [ ] Dataset downloaded and loaded correctly, basic graph stats printed/logged
- [ ] IC model implemented and tested on a small toy graph before running on full data
- [ ] LT model implemented and tested on a small toy graph before running on full data
- [ ] Greedy algorithm implemented, correctly using Monte Carlo spread estimation
- [ ] All four centrality heuristics implemented
- [ ] Random baseline implemented
- [ ] Full experiment run across all budget levels (5, 10, 20, 50) for both IC and LT models
- [ ] Results table generated (all methods × budgets × both diffusion models)
- [ ] All required plots generated
- [ ] Random seeds fixed and documented for reproducibility
- [ ] README written explaining how to reproduce results
- [ ] Code is clean enough to be read and understood by someone writing a paper around it (clear variable names, comments explaining the "why" for anything non-obvious)

## 8. Notes on downstream use

Once this pipeline runs and produces results, those results (tables and figures) will be used to write the actual research paper separately. Claude Code's job ends at producing correct, reproducible, well-documented results and code, it does not need to write the paper text itself. If anything in the results comes out surprising or ambiguous (e.g. a heuristic outperforming greedy at some budget level, which can genuinely happen due to Monte Carlo variance or specific graph structure), flag it clearly in the README or run output rather than silently smoothing it over. That kind of finding is often the most interesting part of a paper like this.
