from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import pandas as pd
import networkx as nx

from src.config import DATASETS, resolve_data_path
from src.utils.io import load_snap_graph
from src.utils.motifs import count_motif_signatures
from src.algorithms.rand_esu import (
    RandESUParams,
    rand_esu_sample,
    parallel_rand_esu_sample,
    parallel_esu_count,
    parallel_esu_enumerate_with_signatures,
    esu_enumerate,
)
from src.experiments.common import build_p_schedule
from src.utils.randomize import randomize_graph_degree_preserving


def run_sampling(G: nx.Graph, k: int, q: float, schedule: str, seed: int):
    import random as pyrand

    pyrand.seed(seed)
    p_depth = build_p_schedule(k, q, schedule)
    params = RandESUParams(k=k, p_depth=p_depth)
    samples_iter = rand_esu_sample(G, params)
    total, freq = count_motif_signatures(G, samples_iter)
    return total, freq


def _worker_edge_swap_iter(
    G: nx.Graph,
    k: int,
    q: float,
    schedule: str,
    seed: int,
    swaps_per_edge: int,
    tries_per_swap: int,
    full_enumeration: bool = False,
) -> tuple[int, float, dict]:
    """Worker function for a single edge-swap iteration.

    Aligned with FANMODPlus C++ implementation:
    - Randomizes graph using degree-preserving edge swaps
    - Either fully enumerates or samples subgraphs
    - Returns frequency dictionary for Z-score calculation
    """
    # Randomize graph (matching C++ num_exchanges and num_tries)
    H = randomize_graph_degree_preserving(
        G,
        swaps_per_edge=swaps_per_edge,
        tries_per_swap=tries_per_swap,
        seed=seed,
    )

    if full_enumeration:
        # Full enumeration (q=1.0) as done in article for accuracy
        from src.algorithms.rand_esu import esu_enumerate
        from src.utils.motifs import count_motif_signatures

        samples_iter = esu_enumerate(H, k)
        total, freq = count_motif_signatures(H, samples_iter)
    else:
        # Sampling mode
        total, freq = run_sampling(H, k, q, schedule, seed)

    return seed, total, freq


def _worker_original_sample(
    G: nx.Graph,
    k: int,
    q: float,
    schedule: str,
    seed: int,
) -> tuple[int, int, dict]:
    """Worker function for sampling the original graph."""
    total, freq = run_sampling(G, k, q, schedule, seed)
    return seed, total, freq


def _load_rand_esu_results_from_csv(
    output_dir: Path,
    dataset: str,
    k: int,
    q: float,
    schedule: str,
    seeds: List[int],
) -> Optional[Dict[str, float]]:
    """Load existing RAND-ESU results from CSV files and compute average concentrations.

    Returns:
        Dictionary mapping signature (str) -> average concentration, or None if files not found.
    """
    original_concentrations: Dict[str, List[float]] = {}

    for seed in seeds:
        csv_path = output_dir / dataset / f"k{k}" / f"q{q}_{schedule}_seed{seed}.csv"
        if not csv_path.exists():
            print(f"  [WARNING] RAND-ESU CSV not found: {csv_path}")
            return None

        try:
            # Force 'signature' column to be read as string to preserve leading zeros
            df = pd.read_csv(csv_path, dtype={"signature": str})
            for _, row in df.iterrows():
                sig = row["signature"]
                conc = row["concentration"]
                original_concentrations.setdefault(sig, []).append(conc)
            print(
                f"  ✓ Loaded RAND-ESU seed={seed} from {csv_path.name}: {len(df)} motif classes",
                flush=True,
            )
        except Exception as e:
            print(f"  [ERROR] Failed to read {csv_path}: {e}")
            return None

    # Compute average across seeds
    avg_original = {
        sig: statistics.mean(vals) for sig, vals in original_concentrations.items()
    }
    return avg_original


def run_edge_swap_significance(
    dataset: str,
    data_dir: Path,
    output_dir: Path,
    k: int = 3,
    q: float = 1.0,
    schedule: str = "uniform",
    seeds: Optional[Iterable[int]] = None,
    random_graphs: int = 1000,
    swaps_per_edge: Optional[int] = None,
    tries_per_swap: int = 3,
    max_nodes: Optional[int] = None,
    use_existing_rand_esu: bool = True,
    full_enumeration: bool = True,
    max_workers: Optional[int] = None,
) -> Dict[str, object]:
    """Run degree-preserving ensemble significance and persist artifacts.

    Aligned with Wernicke (2005) Table 2 and FANMODPlus C++ implementation:
    - Article used 10,000 random graphs for Table 2 results
    - C++ default: 1000 random graphs (num_r_nets)
    - C++ default switch factor: 100 (directed), 10 (undirected)

    Args:
        dataset: Dataset name from config
        data_dir: Path to data directory
        output_dir: Path to output directory
        k: Subgraph size (default 3 for triads)
        q: Sampling probability (1.0 = full enumeration, as in article)
        schedule: Sampling schedule ("uniform" for full enumeration)
        seeds: Random seeds for sampling
        random_graphs: Number of random graphs to generate (article: 10000, C++ default: 1000)
        swaps_per_edge: Switch factor for edge swaps (default: 100 directed, 10 undirected)
        tries_per_swap: Retained for API compatibility (unused by mfinder-style swaps)
        max_nodes: Optional limit on nodes to load
        use_existing_rand_esu: If True, load original concentrations from existing CSV files
        full_enumeration: If True, use full enumeration instead of sampling (recommended for k=3)
        max_workers: Optional cap on parallel workers for edge-swap ensemble

    Note:
        Currently only k=3 is supported. Higher k values will be overridden to k=3
        with a warning, as the significance analysis is designed for triad patterns.
    """

    # Guard: Only k=3 is supported for significance analysis
    if k != 3:
        print(
            f"[WARNING] Edge-swap significance only supports k=3 (triads). "
            f"Requested k={k} will be overridden to k=3."
        )
        k = 3

    seed_list = list(seeds) if seeds is not None else [1, 2, 3]
    cfg = DATASETS.get(dataset)
    if cfg is None:
        raise SystemExit(f"Unknown dataset key {dataset}")
    path = resolve_data_path(data_dir, dataset)
    G = load_snap_graph(path, directed=cfg.directed, max_nodes=max_nodes)
    if swaps_per_edge is None:
        swaps_per_edge = 100 if cfg.directed else 10

    print(
        f"Loaded graph {dataset}: n={G.number_of_nodes()} m={G.number_of_edges()} directed={cfg.directed}"
    )

    # Try to load existing RAND-ESU results if requested
    avg_original = None
    if use_existing_rand_esu:
        print(f"Loading original concentrations from existing RAND-ESU CSV files...")
        avg_original = _load_rand_esu_results_from_csv(
            output_dir=output_dir,
            dataset=dataset,
            k=k,
            q=q,
            schedule=schedule,
            seeds=seed_list,
        )
        if avg_original is not None:
            print(f"  ✓ Loaded {len(avg_original)} motif classes from existing results")
        else:
            print(f"  [FALLBACK] Will re-run sampling (existing CSVs not found)")

    # Fall back to re-running enumeration/sampling if needed
    if avg_original is None:
        original_runs = []

        if full_enumeration:
            # Full enumeration (as done in article) - only need one run
            print(f"Enumerating original graph (full enumeration, k={k})...")
            from src.algorithms.rand_esu import esu_enumerate
            from src.utils.motifs import count_motif_signatures

            t_start_enum = time.time()
            samples_iter = esu_enumerate(G, k)
            total, freq = count_motif_signatures(G, samples_iter)
            t_end_enum = time.time()
            original_runs.append((1, total, freq))
            print(
                f"  ✓ Enumerated {total} subgraphs, {len(freq)} motif classes in {t_end_enum - t_start_enum:.2f}s",
                flush=True,
            )
        else:
            # Sampling mode with multiple seeds
            print(
                f"Sampling original graph with {len(seed_list)} seeds (sequential)..."
            )
            for seed in seed_list:
                t_seed_start = time.time()
                total, freq = run_sampling(G, k, q, schedule, seed)
                t_seed_end = time.time()
                original_runs.append((seed, total, freq))
                print(
                    f"  ✓ Original seed={seed}: total={total} motif_classes={len(freq)} in {t_seed_end - t_seed_start:.2f}s",
                    flush=True,
                )

        original_concentrations: Dict[str, List[float]] = {}
        for seed, total, freq in original_runs:
            for sig, cnt in freq.items():
                original_concentrations.setdefault(sig, []).append(
                    cnt / total if total else 0.0
                )
        avg_original = {
            sig: statistics.mean(vals) for sig, vals in original_concentrations.items()
        }

    ensemble_concentrations = {sig: [] for sig in avg_original.keys()}

    t_start = time.time()

    # Dynamic worker allocation: use all cores but leave 2 for system/other tasks
    # Ensure at least 1 worker. Allow optional cap.
    if max_workers is None:
        max_workers = max(1, (cpu_count() or 4) - 2)
    else:
        max_workers = max(1, max_workers)
    print(
        f"Starting parallel edge-swap significance with {max_workers} workers for {random_graphs} graphs..."
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(random_graphs):
            seed = seed_list[i % len(seed_list)] + 1000
            futures.append(
                executor.submit(
                    _worker_edge_swap_iter,
                    G,
                    k,
                    q,
                    schedule,
                    seed,
                    swaps_per_edge,
                    tries_per_swap,
                    full_enumeration,
                )
            )

        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            try:
                seed_res, total, freq = future.result()
                for sig in avg_original.keys():
                    # Both avg_original and freq have string keys (signature strings with leading zeros)
                    c = (freq.get(sig, 0) / total) if total else 0.0
                    ensemble_concentrations[sig].append(c)
            except Exception as e:
                print(f"    [EdgeSwap] Worker failed: {e}")
                continue

            t_now = time.time()
            elapsed = t_now - t_start
            # Estimate ETA based on average time per completed graph
            avg_time = elapsed / completed_count
            eta = avg_time * (random_graphs - completed_count)

            print(
                f"    [EdgeSwap {completed_count}/{random_graphs}] "
                f"elapsed={elapsed:.1f}s ETA={eta:.1f}s",
                flush=True,
            )

    rows = []
    for sig, orig_c in avg_original.items():
        ens = ensemble_concentrations.get(sig, [])
        if ens and len(ens) > 1:
            mean_ens = statistics.mean(ens)
            # Use sample standard deviation (N-1) as in C++ calc_deviation
            stdev_ens = statistics.stdev(ens)  # stdev uses N-1, pstdev uses N
            z = (orig_c - mean_ens) / stdev_ens if stdev_ens > 0 else float("nan")
            # P-value: fraction of random graphs where concentration > original
            # Matches C++ calculation exactly
            p_value = sum(1 for c in ens if c > orig_c) / len(ens)
        else:
            mean_ens = 0.0
            stdev_ens = 0.0
            z = float("nan")
            p_value = float("nan")
        if mean_ens > 0:
            ratio = orig_c / mean_ens
        elif orig_c > 0:
            ratio = float("inf")
        else:
            ratio = float("nan")
        rows.append(
            {
                "dataset": dataset,
                "k": k,
                "signature": sig,
                "orig_concentration": orig_c,
                "rand_mean_concentration": mean_ens,
                "rand_std_concentration": stdev_ens,
                "enrichment_ratio": ratio,
                "z_score": z,
                "p_value": p_value,
                "q": q,
                "schedule": schedule,
                "directed": cfg.directed,
                "random_graphs": random_graphs,
                "swaps_per_edge": swaps_per_edge,
                "tries_per_swap": tries_per_swap,
                "full_enumeration": full_enumeration,
                "seeds": seed_list,
            }
        )

    df = pd.DataFrame(rows)
    subdir = output_dir / dataset / f"k{k}" / "significance"
    subdir.mkdir(parents=True, exist_ok=True)
    out_csv = subdir / f"significance_q{q}_{schedule}.csv"
    df.to_csv(out_csv, index=False)

    meta = {
        "dataset": dataset,
        "k": k,
        "q": q,
        "schedule": schedule,
        "seeds": seed_list,
        "random_graphs": random_graphs,
        "swaps_per_edge": swaps_per_edge,
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "max_nodes": max_nodes,
        "ratio_definition": "orig_concentration / rand_mean_concentration",
        "z_score_note": (
            "Z-scores are provided for reference; Wernicke (2005) reports ratios "
            "of observed to mean random concentrations. Z can be extreme when "
            "rand_std_concentration is tiny."
        ),
    }
    meta_path = subdir / f"significance_q{q}_{schedule}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    try:
        top = df.sort_values("z_score", ascending=False).head(10)
        txt = subdir / f"significance_q{q}_{schedule}.txt"
        with open(txt, "w") as fsum:
            fsum.write("Significance via edge swaps (degree-preserving ensemble)\n")
            fsum.write(
                f"Dataset={dataset} k={k} q={q} schedule={schedule} directed={cfg.directed}\n"
            )
            fsum.write(
                f"random_graphs={random_graphs} swaps_per_edge={swaps_per_edge} seeds={seed_list}\n\n"
            )
            fsum.write(
                "Metrics: enrichment_ratio=orig_concentration/rand_mean_concentration (paper-aligned). "
                "Z-scores are included for reference and can be extreme when rand_std_concentration is tiny.\n"
            )
            top_ratio = df.sort_values("enrichment_ratio", ascending=False).head(10)
            fsum.write("Top-10 motifs by enrichment ratio (orig vs ensemble mean):\n")
            for _, r in top_ratio.iterrows():
                fsum.write(
                    f"  sig={r['signature']} ratio={r['enrichment_ratio']:.6g} orig={r['orig_concentration']:.6g} "
                    f"rand_mean={r['rand_mean_concentration']:.6g}\n"
                )
            fsum.write("\nTop-10 motifs by z-score (orig vs ensemble):\n")
            for _, r in top.iterrows():
                fsum.write(
                    f"  sig={r['signature']} z={r['z_score']:.3f} orig={r['orig_concentration']:.6g} "
                    f"rand_mean={r['rand_mean_concentration']:.6g} rand_std={r['rand_std_concentration']:.6g}\n"
                )
    except Exception:
        pass

    return {
        "df": df,
        "csv_path": out_csv,
        "meta_path": meta_path,
        "subdir": subdir,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate motif significance via degree-preserving randomized graphs.\n"
        "Aligned with Wernicke (2005) and FANMODPlus C++ implementation."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--k", type=int, default=3, help="Subgraph size (default: 3)")
    parser.add_argument(
        "--q",
        type=float,
        default=1.0,
        help="Sampling probability (1.0=full enumeration, default: 1.0)",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="uniform",
        help="Sampling schedule (default: uniform for full enumeration)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[1],
        help="Random seeds (default: [1] for full enumeration)",
    )
    parser.add_argument(
        "--random-graphs",
        type=int,
        default=1000,
        help="Number of randomized graphs (article: 10000, C++ default: 1000)",
    )
    parser.add_argument(
        "--swaps-per-edge",
        type=int,
        default=None,
        help="Switch factor for edge swaps (default: 100 directed, 10 undirected)",
    )
    parser.add_argument(
        "--tries-per-swap",
        type=int,
        default=3,
        help="Tries per swap attempt (C++ default: 3)",
    )
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--no-use-existing",
        action="store_true",
        help="Force re-running enumeration instead of loading from existing CSVs",
    )
    parser.add_argument(
        "--no-full-enumeration",
        action="store_true",
        help="Use sampling instead of full enumeration (not recommended for k=3)",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = run_edge_swap_significance(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        k=args.k,
        q=args.q,
        schedule=args.schedule,
        seeds=args.seeds,
        random_graphs=args.random_graphs,
        swaps_per_edge=args.swaps_per_edge,
        tries_per_swap=args.tries_per_swap,
        max_nodes=args.max_nodes,
        use_existing_rand_esu=not args.no_use_existing,
        full_enumeration=not args.no_full_enumeration,
    )

    print(f"Saved significance results: {result['csv_path']}")


if __name__ == "__main__":
    main()
