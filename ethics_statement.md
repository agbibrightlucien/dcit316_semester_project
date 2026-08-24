# Ethics Statement

I used the publicly released SNAP ego-Facebook dataset (facebook_combined.txt) to benchmark seed selection algorithms for influence maximization under a simulated information diffusion model. I did not collect any new data from human subjects.

## Data privacy
- The dataset consists of anonymized integer node identifiers only. No names, profile content, messages, or other personally identifying information is included or used anywhere in my project.
- No attempt is made to re-identify, de-anonymize, or link any node back to a real individual.
- The dataset was originally collected by SNAP from consenting participants via a Facebook survey application and has been publicly released for research use since 2012 (McAuley & Leskovec, 2012); my project is downstream, secondary use of that already-public, already-anonymized release.

## Intended use and framing
- The "influence maximization" framing in my project (and in the foundational Kempe, Kleinberg & Tardos 2003 paper it builds on) is explicitly academic and benchmarking-oriented: comparing the computational efficiency and solution quality of different seed-selection algorithms against a well-studied theoretical baseline (the greedy algorithm).
- The marketing seeding framing I used to motivate this project is a standard, widely used pedagogical framing in the influence maximization literature. I used it only to motivate the budget-constrained research question, not to design or deploy an actual targeting or marketing system.
- None of the results, code, or artifacts from my project are intended for use in real-world targeted marketing, political messaging, coordinated influence campaigns, or any other application involving real people's data. This is a coursework/research benchmark using a static, anonymized academic dataset.

## Reproducibility and transparency
All randomness in my project (Monte Carlo diffusion simulations, the random-seed baseline, betweenness centrality's source sampling) uses fixed, documented random seeds (see `README.md` and `scripts/run_full_analysis.py`), so results are independently verifiable rather than being presented as unfalsifiable claims.
