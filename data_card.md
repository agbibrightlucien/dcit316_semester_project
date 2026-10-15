# Data Card: SNAP ego-Facebook

## Source
- **Original source:** [SNAP ego-Facebook](http://snap.stanford.edu/data/ego-Facebook.html), Stanford Network Analysis Project.
- **Mirror I used:** [github.com/jcatw/snap-facebook](https://github.com/jcatw/snap-facebook), which commits `facebook_combined.txt` directly. I used this mirror instead of the original SNAP page and confirmed it was the identical file by comparing checksums.
- **Citations:**
  - Leskovec, J., & Krevl, A. (2014). "SNAP Datasets: Stanford Large Network Dataset Collection." http://snap.stanford.edu/data.
  - McAuley, J., & Leskovec, J. (2012). "Learning to Discover Social Circles in Ego Networks." NIPS 2012.

## What it is
This is a single undirected social graph formed by combining 10 "ego networks" from Facebook, the graph of an anonymized user and the connections among their friends, collected via a Facebook app from consenting survey participants.

## Stats
- **Nodes:** 4,039, which I verified myself using `src/graph_utils.load_graph` and `compute_basic_stats` when loading the data.
- **Edges:** 88,234
- **Type:** undirected, unweighted, simple graph
- **Connectivity:** a single connected component (verified at load time; see `reports/graph_stats.json`)
- **Average degree:** ~43.7
- **Density:** ~0.0108
- **Average clustering coefficient:** ~0.61 (high, as expected for ego-networks: friend groups form dense, overlapping cliques)

## Format
Plain-text edge list, one edge per line, whitespace-separated integer node
IDs, e.g. `0 1` means an (undirected) edge between node 0 and node 1.

## Node identity and anonymization
Node IDs are anonymized integers assigned by the original SNAP release; they
carry no recoverable personal information (no names, handles, or profile
data). I did not use any node attributes or features, such as the feature_map.txt file included in some SNAP ego network releases. I only used the raw edge list, so no potentially sensitive per user attribute data is touched at all.

## License / research use
Distributed by SNAP for research/educational use, and mirrored verbatim in the `jcatw/snap-facebook` GitHub repository I used. I am using this dataset only for algorithmic benchmarking (comparing seed selection methods under simulated information diffusion); I made no attempt to re-identify individuals or use the data for any purpose other than my stated research question.

## Known limitations
- **Static snapshot:** the graph is a fixed snapshot with no timestamps, so
  temporal diffusion dynamics (e.g. real posting/sharing times) cannot be
  modeled, IC/LT simulations here are purely structural.
- **Small by modern social-network standards:** at ~4,000 nodes this is a
  useful testbed for exact/near-exact algorithms (greedy is computationally
  tractable here) but results should not be assumed to generalize to
  graphs with millions of nodes without further study.
- **Single platform, single collection method:** all nodes come from
  Facebook ego-networks collected via one survey app, so structural
  properties (e.g. the unusually high clustering coefficient) may not
  transfer to other platforms.
- **No edge weights / interaction strength:** all edges are treated as
  equally likely to transmit influence within a given diffusion model
  parameterization (see `README.md` for the exact IC/LT parameters used);
  the dataset itself provides no signal to differentiate strong vs. weak
  ties.
