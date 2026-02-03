# RAND-ESU Motif Sampling Project

This `src/` package provides a modular Python implementation of RAND-ESU for sampling connected induced subgraphs (motifs) on SNAP datasets, plus experiment runners and utilities.

## 1. Modules

### 1.1 Core Configuration
- `src/config.py`: Dataset mapping (file names, directedness) and path resolvers

### 1.2 Utilities (`src/utils/`)
- `src/utils/io.py`: SNAP edge-list loader (supports `.txt` or `.gz`, optional node cap)
- `src/utils/motifs.py`: Canonical motif signatures via brute-force minimal adjacency string (k <= 6)
- `src/utils/randomize.py`: Degree-preserving edge swap utilities for random graph generation
- `src/utils/visualize.py`: Motif visualization and plotting utilities

### 1.3 Algorithms (`src/algorithms/`)
- `src/algorithms/rand_esu.py`: ESU and RAND-ESU implementations (sampling with depth-wise probabilities)
- `src/algorithms/esa.py`: Edge Sampling Algorithm (ESA) baseline implementation

### 1.4 Experiment Runners (`src/experiments/`)
- `src/experiments/run_rand_esu.py`: CLI runner to sample motifs and export results (CSV + JSON)
- `src/experiments/run_all_datasets_parallel.py`: Parallel orchestrator (multi-process) with intra-run and inter-run parallelism (recommended)
- `src/experiments/common.py`: Shared utilities (depth-wise probability builders, parameter handling)
- `src/experiments/baselines.py`: Baseline algorithm implementations and comparison utilities

### 1.5 Benchmarking & Analysis
- `src/experiments/benchmark_speed_quality.py`: Benchmarks speed (Fig 3a-style) and quality (Fig 3b-style) vs ESA baseline
- `src/experiments/compute_basic_stats.py`: Quick graph stats (n, m, WCC/SCC sizes)
- `src/experiments/compute_network_metrics.py`: Comprehensive network metrics (degree distribution, clustering, centrality)
- `src/experiments/network_properties_table.py`: Network properties table generation for reports
- `src/experiments/aggregate_results.py`: Results aggregation, correlation analysis, cross-dataset motif analysis
- `src/experiments/plot_from_results.py`: Generate plots from saved experiment results
- `src/experiments/validate_sampling_unbiasedness.py`: Validate RAND-ESU sampling unbiasedness

### 1.6 Significance Testing
- `src/experiments/significance_edge_swaps.py`: Significance via degree-preserving random graph ensemble (edge swaps)
- `src/experiments/significance_direct_bender_canfield.py`: Bender-Canfield method for significance testing

- `src/experiments/compare_significance_methods.py`: Assembles ensemble baseline summaries (Table-2-like reference setup)

## 2. Quickstart

1. (Optional) Create a virtual environment.
2. Install dependencies from the repo root:

```
pip install -r requirements.txt
```

3. Run a small smoke test on Wiki-Vote (limits nodes for speed):

```
python -m src.experiments.run_rand_esu --datasets Wiki-Vote --k 3 --q 0.001 --max-nodes 5000 --schedule fine --seed 1 --data-dir data --output-dir results
```

This will create `results/Wiki-Vote_k3_q0.001_fine_seed1.csv` and a corresponding meta JSON.

## 3. Running Experiments (Parallel Runner)

The `run_all_datasets_parallel.py` script is the recommended way to run comprehensive experiments. It supports both CLI arguments and interactive prompts.

### 3.1 Available Datasets

The following datasets are configured in `src/config.py` and should be placed in the `data/` directory:

| Dataset | File | Type | Description |
|---------|------|------|-------------|
| Amazon0302 | `Amazon0302.txt` | Directed | Amazon product co-purchasing network |
| CA-AstroPh | `CA-AstroPh.txt` | Undirected | ArXiv Astro Physics co-authorship |
| Wiki-Vote | `Wiki-Vote.txt` | Directed | Wikipedia adminship vote network |

Download datasets from [SNAP](https://snap.stanford.edu/data/) and place the edge-list `.txt` files in `data/`.

### 3.2 Experiment Configuration

Our current run (k=3 only) uses the following parameters:

| k | q | Schedule | Child Selection | Baseline Samples | Significance |
|---|---|----------|-----------------|------------------|--------------|
| 3 | 0.1 | fine | bernoulli | 5k | Edge-swap + Direct BC (directed only) |

**Notes:**
- Max-degree filtering is disabled (matches the Wernicke 2005 paper defaults).
- Significance testing is only run for directed k=3 triads.
- k=4 reruns are complete (results_k4); k=5 runs are currently in progress.

### 3.3 Running via CLI

**k=3 run (current results):**
```bash
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

### 3.4 Running via Interactive Mode

For interactive configuration, simply run without `--no-interactive`:

```bash
python -m src.experiments.run_all_datasets_parallel
```

The script will prompt for each parameter with sensible defaults:

```
[interactive] Press Enter to accept defaults in brackets.
(Defaults are based on the Wernicke 2005 paper recommendations)

Datasets (comma-separated or 'all') [Amazon0302,CA-AstroPh,Wiki-Vote,...]: Amazon0302,CA-AstroPh,Wiki-Vote
k sizes (comma-separated) [3,4,5,6]: 3,4,5
q values (sampling fraction, 0.1=10%) [0.1]: 0.1
schedule (fine|coarse|geometric|skewed - fine is recommended) [fine]: 
child-selection (bernoulli|balanced) [bernoulli]: 
seeds (multiple for variance estimation) [1,2,3]: 
max-workers (parallel processes) [<CPU count>]: 
max-nodes (None for all nodes, or integer for testing) [None]: 
memory-optimized counting (recommended for k>=4)? (y/n) [y]: 
data-dir [data]: 
output-dir [results]: results_k3
baseline (none|esa) [esa]: 
baseline samples per run [2000]: 5000
baseline max retries [40]: 
baseline plots? (y/n) [y]: 
significance method (none|edge-swap|direct|both) [direct]: both
```

**Interactive tips:**
- Press Enter to accept the default value shown in brackets
- Use `both` for significance method to get edge-swap ensemble and direct BC comparison
- Memory-optimized mode is recommended for k≥4 to reduce RAM usage

### 3.5 CLI Options Reference

| Option | Default | Description |
|--------|---------|-------------|
| `--datasets` | All 4 | Space-separated dataset keys |
| `--k` | 3 4 5 6 | Motif sizes to enumerate |
| `--q` | 0.1 | Sampling fraction(s) |
| `--schedule` | fine | Depth probability schedule |
| `--child-selection` | bernoulli | Child node selection strategy |
| `--seed` | 1 2 3 | Random seeds for variance estimation |
| `--max-workers` | CPU count | Parallel worker processes |
| `--max-nodes` | None | Node cap for testing |
| `--memory-optimized` | True | Use memory-lean counting |
| `--baseline` | esa | Baseline algorithm (none\|esa) |
| `--baseline-samples` | 2000 | ESA samples per run |
| `--significance-method` | direct | none\|edge-swap\|direct\|both |
| `--significance-k` | 3 | Motif size for significance |
| `--no-interactive` | False | Skip interactive prompts |
| `--data-dir` | data | Input data directory |
| `--output-dir` | results | Output results directory |

### 3.6 Output Structure

Results are organized as:
```
results_k3/
├── performance_scaling_parallel.csv
├── run_config.json
├── Amazon0302/
│   ├── summary.txt
│   ├── summary_results.txt
│   ├── k3/
│   │   ├── q0.1_fine_seed1.csv
│   │   ├── q0.1_fine_seed1_meta.json
│   │   ├── plots/
│   │   ├── baseline/esa/
│   │   ├── significance/
│   │   ├── significance_direct_bc/
│   │   └── significance_compare/
├── CA-AstroPh/
└── Wiki-Vote/
```

## 4. Reproducibility

- Environments: Per-dataset `summary.txt` files include Python version, platform, parameters, and per-k aggregated stats. Each run writes CSV results, meta JSON (including `p_depth`, `seed`, `runtime_sec`, and `realized_fraction` when feasible), and plots.
- Randomness: Set seeds via `--seed`. For multi-seed experiments, results and per-dataset summaries reflect all seeds used.

