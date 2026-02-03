<p align="center">
  <img src="assets/Screenshot_555.png" alt="Motif examples" width="420">
</p>

# RAND-ESU Motif Sampling

This repository contains a Python implementation of RAND-ESU for sampling
connected induced subgraphs (motifs) in large networks. It follows the
depth-wise probabilistic pruning approach described by Wernicke (2005) and
includes ESA baselines for comparison, parallel execution, and significance
testing workflows.

## Overview
- Unbiased motif sampling via depth-wise probabilities (RAND-ESU).
- ESA baseline with bias correction for comparison.
- Parallel execution for large graphs.
- Significance testing with edge-swap ensembles and direct expectations.

## Repository Structure
- `src/algorithms/`
  - `rand_esu.py`: ESU and RAND-ESU implementations.
  - `esa.py`: Edge Sampling Algorithm (ESA) baseline.
- `src/utils/`
  - `io.py`: SNAP edge-list loader.
  - `motifs.py`: Canonical motif signatures (k <= 6).
  - `randomize.py`: Degree-preserving edge swaps.
  - `visualize.py`: Plotting utilities.
- `src/experiments/`
  - `run_rand_esu.py`: CLI runner for a single dataset.
  - `run_all_datasets_parallel.py`: Parallel runner (recommended).
  - `aggregate_results.py`: Aggregation and correlation analysis.
  - `benchmark_speed_quality.py`: Runtime and quality benchmarks.
  - `significance_edge_swaps.py`: Ensemble-based significance.
  - `significance_direct_bender_canfield.py`: Direct expectations.

## Setup
1. (Optional) Create and activate a virtual environment.
2. Install dependencies:

```
pip install -r requirements.txt
```

## Quickstart
Run a small smoke test on Wiki-Vote (node cap for speed):

```
python -m src.experiments.run_rand_esu \
  --datasets Wiki-Vote \
  --k 3 \
  --q 0.001 \
  --max-nodes 5000 \
  --schedule fine \
  --seed 1 \
  --data-dir data \
  --output-dir results
```

This creates `results/Wiki-Vote_k3_q0.001_fine_seed1.csv` and a meta JSON.

## Datasets
Place SNAP edge-list files in `data/`:

| Dataset | File | Type |
|---------|------|------|
| Amazon0302 | `Amazon0302.txt` | Directed |
| CA-AstroPh | `CA-AstroPh.txt` | Undirected |
| Wiki-Vote | `Wiki-Vote.txt` | Directed |
| roadNet-CA | `roadNet-CA.txt` | Undirected |

Download from https://snap.stanford.edu/data/.

## Running Full Experiments
The parallel runner supports CLI and interactive modes.

Example for k=3:

```
python -m src.experiments.run_all_datasets_parallel \
  --datasets Wiki-Vote Amazon0302 CA-AstroPh \
  --k 3 \
  --q 0.1 \
  --schedule fine \
  --child-selection bernoulli \
  --seed 1 2 3 \
  --baseline esa \
  --baseline-samples 5000 \
  --significance-method both \
  --significance-random-graphs 100 \
  --significance-full-enumeration \
  --direct-T 100000 \
  --include-lambda-correction \
  --memory-optimized \
  --no-interactive \
  --data-dir data \
  --output-dir results_k3
```

Interactive mode:

```
python -m src.experiments.run_all_datasets_parallel
```

## Output Layout
Results are grouped by dataset and motif size, for example:

```
results_k3/
├── performance_scaling_parallel.csv
├── run_config.json
├── Amazon0302/
│   ├── summary.txt
│   ├── summary_results.txt
│   └── k3/
│       ├── q0.1_fine_seed1.csv
│       ├── q0.1_fine_seed1_meta.json
│       ├── plots/
│       ├── baseline/
│       └── significance/
└── Wiki-Vote/
```

## Notes
- Directed graphs use weak connectivity during ESU expansion to reach
  in-star motifs; edge directions are preserved in motif classification.
- Significance testing is typically run for directed k=3 triads.
- For k >= 4, `--memory-optimized` is recommended.

## References
- Wernicke, S. (2005). A Faster Algorithm for Detecting Network Motifs.
- Kashtan, N. et al. (2004). Efficient Sampling Algorithm for Estimating
  Subgraph Concentrations and Detecting Network Motifs.
