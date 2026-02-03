from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import json
import time
from typing import Iterable, List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import random

import pandas as pd

import platform
from src.config import DATASETS, resolve_data_path
from src.utils.io import load_snap_graph
from src.utils.motifs import count_motif_signatures, triad_label_from_signature
from src.algorithms.rand_esu import (
    RandESUParams,
    rand_esu_sample,
    parallel_rand_esu_sample,
    parallel_rand_esu_count,
    parallel_esu_count,
    esu_enumerate,
)
from src.experiments.baselines import run_esa_baseline, generate_overlay
from src.experiments.common import build_p_schedule
from src.experiments.significance_edge_swaps import run_edge_swap_significance
from src.utils.visualize import (
    plot_motif_distribution,
    plot_motif_distribution_horizontal,
    plot_seed_boxplot,
    plot_scatter_xy,
    plot_significance_scatter,
)


def calculate_worker_allocation(total_cores, total_experiments):
    """
    Dynamically calculate optimal worker allocation based on available cores and experiments.

    Strategy:
    - If cores >= 6×experiments: Use 6 cores per run (maximum intra-parallelism)
    - If cores >= 5×experiments: Use 5 cores per run
    - If cores >= 4×experiments: Use 4 cores per run
    - If cores >= 3×experiments: Use 3 cores per run
    - If cores >= 2×experiments: Use 2 cores per run
    - Otherwise: Use 1 core per run (maximize parallel runs)

    This ensures we always use all available cores efficiently while applying
    progressively more intra-parallelism when we have extra capacity.

    Args:
        total_cores: Number of CPU cores available
        total_experiments: Number of experiments to run for this k-value

    Returns:
        (max_concurrent_runs, cores_per_run)
    """
    # Calculate how many cores we can give each experiment
    cores_ratio = total_cores / total_experiments

    if cores_ratio >= 6:
        # Abundant cores: use 6 per run for maximum intra-parallelism
        cores_per_run = 6
    elif cores_ratio >= 5:
        # Very high surplus: use 5 per run
        cores_per_run = 5
    elif cores_ratio >= 4:
        # High surplus: use 4 per run
        cores_per_run = 4
    elif cores_ratio >= 3:
        # Good surplus: use 3 per run
        cores_per_run = 3
    elif cores_ratio >= 2:
        # Moderate surplus: use 2 per run
        cores_per_run = 2
    else:
        # Limited cores: maximize parallel runs with 1 core each
        cores_per_run = 1

    # Calculate how many runs can execute concurrently
    max_concurrent_runs = max(1, total_cores // cores_per_run)

    return max_concurrent_runs, cores_per_run


def run_single_experiment(args_tuple):
    """Run a single (k, q, seed) experiment. Designed to be called in parallel."""
    (
        ds_key,
        path,
        directed,
        k,
        q,
        schedule,
        child_selection,
        seed,
        output_dir,
        max_nodes,
        cores_per_run,  # Future use for intra-algorithm parallelism
        max_degree,  # Optional degree filter
        memory_optimized,  # Use memory-lean counting path
    ) = args_tuple

    run_id = f"{ds_key} | k={k} q={q} seed={seed}"

    # Each process loads its own copy of the graph
    print(f"    [{run_id}] Loading graph...", flush=True)
    t_load0 = time.time()
    # Note: Graph loaded without max_degree filtering here
    # max_degree filtering is applied during sampling in RandESUParams
    G = load_snap_graph(path, directed=directed, max_nodes=max_nodes)
    t_load1 = time.time()
    print(
        f"    [{run_id}] Loaded: n={G.number_of_nodes():,} m={G.number_of_edges():,} in {t_load1 - t_load0:.2f}s",
        flush=True,
    )

    # Build parameters
    p_depth = build_p_schedule(k, q, schedule)
    random.seed(seed)
    params = RandESUParams(
        k=k, p_depth=p_depth, child_selection=child_selection, max_degree=max_degree
    )

    # Sample and count - use parallel version if multiple cores allocated
    if cores_per_run > 1:
        t_sample0 = time.time()

        # Progress tracking for chunks with detailed timing
        chunk_progress = {
            "completed": 0,
            "total": 0,
            "start_time": t_sample0,
            "chunk_times": [],
            "last_print": 0,
            "chunk_info": None,
        }

        def progress_callback(
            completed, total, elapsed_sec, chunk_time_sec, chunk_info=None
        ):
            # Handle initial chunk info
            if chunk_info is not None:
                chunk_progress["chunk_info"] = chunk_info
                print(
                    f"    [{run_id}] Sampling with {cores_per_run} cores | {chunk_info['num_chunks']} chunks | ~{chunk_info['avg_chunk_size']} roots/chunk ({chunk_info['total_roots']} total roots)",
                    flush=True,
                )
                return

            chunk_progress["completed"] = completed
            chunk_progress["total"] = total
            chunk_progress["chunk_times"].append(chunk_time_sec)

            # Calculate statistics
            pct = (completed / total * 100) if total > 0 else 0
            avg_chunk_time = sum(chunk_progress["chunk_times"]) / len(
                chunk_progress["chunk_times"]
            )
            remaining_chunks = total - completed
            eta_sec = avg_chunk_time * remaining_chunks

            # Format ETA
            if eta_sec < 60:
                eta_str = f"{eta_sec:.0f}s"
            elif eta_sec < 3600:
                eta_str = f"{eta_sec / 60:.1f}m"
            else:
                eta_str = f"{eta_sec / 3600:.1f}h"

            # Create progress bar (32 chars wide)
            bar_width = 32
            filled = int(bar_width * completed / total)
            bar = "█" * filled + "░" * (bar_width - filled)

            # Print every 10% or every 5 chunks, whichever is more frequent
            print_interval = max(1, min(5, total // 10))
            if (
                completed - chunk_progress["last_print"] >= print_interval
                or completed == total
            ):
                if completed == total:
                    # Final summary
                    print(
                        f"      └─ [{completed}/{total}] 100% │{bar}│ Completed in {elapsed_sec:.1f}s (avg {avg_chunk_time:.2f}s/chunk)",
                        flush=True,
                    )
                else:
                    # Progress update
                    print(
                        f"      └─ [{completed}/{total}] {pct:.0f}% │{bar}│ {avg_chunk_time:.2f}s/chunk | ETA: {eta_str}",
                        flush=True,
                    )
                chunk_progress["last_print"] = completed

        if memory_optimized:
            total, freq = parallel_rand_esu_count(
                G, params, num_cores=cores_per_run, progress_callback=progress_callback
            )
        else:
            samples_iter = parallel_rand_esu_sample(
                G, params, num_cores=cores_per_run, progress_callback=progress_callback
            )
            total, freq = count_motif_signatures(G, samples_iter)
        t_sample1 = time.time()

        # Print throughput info
        samples_per_sec = (
            total / (t_sample1 - t_sample0) if (t_sample1 - t_sample0) > 0 else 0
        )
        mode_str = "memory-opt" if memory_optimized else "stream"
        print(
            f"    [{run_id}] Sampled: {total:,} samples ({samples_per_sec:.0f} samples/sec), {len(freq)} unique classes in {t_sample1 - t_sample0:.2f}s [{mode_str}]",
            flush=True,
        )
    else:
        print(f"    [{run_id}] Sampling motifs (single-core)...", flush=True)
        t_sample0 = time.time()
        samples_iter = rand_esu_sample(G, params)
        total, freq = count_motif_signatures(G, samples_iter)
        t_sample1 = time.time()
        samples_per_sec = (
            total / (t_sample1 - t_sample0) if (t_sample1 - t_sample0) > 0 else 0
        )
        print(
            f"    [{run_id}] Sampled: {total:,} samples ({samples_per_sec:.0f} samples/sec), {len(freq)} unique classes in {t_sample1 - t_sample0:.2f}s",
            flush=True,
        )

    # Prepare results
    print(f"    [{run_id}] Saving results...", flush=True)
    t_save0 = time.time()
    records = []
    for sig, count in freq.items():
        records.append(
            {
                "dataset": ds_key,
                "directed": directed,
                "k": k,
                "q": q,
                "schedule": schedule,
                "seed": seed,
                "total_samples": total,
                "signature": sig,
                "count": count,
                "concentration": (count / total) if total > 0 else 0.0,
            }
        )

    df = pd.DataFrame.from_records(records)

    # Save results
    subdir = output_dir / ds_key / f"k{k}"
    subdir.mkdir(parents=True, exist_ok=True)

    csv_path = subdir / f"q{q}_{schedule}_seed{seed}.csv"
    df.to_csv(csv_path, index=False)

    # Save metadata
    try:
        rel_file = str(path.resolve().relative_to(Path("data").resolve()))
    except Exception:
        rel_file = path.name

    # Compute realized sampling fraction for k=3 via full enumeration
    # For k>=4, full enumeration is too expensive (billions of subgraphs)
    # Use parallel enumeration for speed on large graphs
    total_subgraphs = None
    realized_fraction = None
    if k == 3:
        try:
            import os

            enum_cores = max(1, min(cores_per_run, (os.cpu_count() or 4)))
            print(
                f"    [{run_id}] Enumerating all k=3 subgraphs (parallel, {enum_cores} cores)...",
                flush=True,
            )
            t_enum0 = time.time()
            all_k = parallel_esu_count(G, k, num_cores=enum_cores)
            t_enum1 = time.time()
            total_subgraphs = all_k
            if all_k > 0:
                # realized_fraction = actual samples / total possible subgraphs
                realized_fraction = total / all_k
            print(
                f"    [{run_id}] Enumerated {all_k:,} subgraphs in {t_enum1 - t_enum0:.2f}s, realized_fraction={realized_fraction:.6f}",
                flush=True,
            )
        except Exception as e:
            print(f"    [{run_id}] Enumeration failed: {e}", flush=True)

    meta = {
        "dataset": ds_key,
        "file": rel_file,
        "directed": directed,
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "k": k,
        "q": q,
        "p_depth": p_depth,
        "child_selection": params.child_selection,
        "schedule": schedule,
        "seed": seed,
        "runtime_sec": t_sample1 - t_sample0,
        "loaded_sec": t_load1 - t_load0,
        "total_samples": total,
        "unique_motif_classes": len(freq),
        "total_subgraphs": total_subgraphs,  # Total k-subgraphs in graph (k=3 only)
        "realized_fraction": realized_fraction,  # samples / total_subgraphs (k=3 only)
        "max_nodes": max_nodes,
    }

    json_path = subdir / f"q{q}_{schedule}_seed{seed}_meta.json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Plot
    plot_path = subdir / f"q{q}_{schedule}_seed{seed}_motifs.png"
    plot_title = f"{ds_key} k={k} q={q} seed={seed} sel={params.child_selection}"
    plot_motif_distribution(freq, total, plot_title, plot_path)

    t_save1 = time.time()
    print(
        f"    [{run_id}] Saved: csv+json+plot in {t_save1 - t_save0:.2f}s", flush=True
    )

    return {
        "ds_key": ds_key,
        "k": k,
        "q": q,
        "seed": seed,
        "total_samples": total,
        "unique_classes": len(freq),
        "runtime_sec": t_sample1 - t_sample0,
        "load_sec": t_load1 - t_load0,
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
    }


def run_dataset_pipeline(
    args: argparse.Namespace,
    ds_key: str,
    dataset_cores: int,
    total_cores: int,
) -> Dict[str, object]:
    """Run the full experiment suite for a single dataset.

    This function encapsulates baseline runs, RAND-ESU sampling, aggregation,
    and significance analysis for one dataset. It is safe to run in parallel
    across datasets.
    """
    max_workers = dataset_cores
    cfg = DATASETS.get(ds_key)
    if cfg is None:
        print(f"[WARN] Unknown dataset key '{ds_key}', skipping")
        return {"ds_key": ds_key, "status": "skipped"}

    path = resolve_data_path(args.data_dir, ds_key)
    print(f"\n{'=' * 70}")
    print(f"Dataset: {ds_key} ({'directed' if cfg.directed else 'undirected'})")
    print(f"Path: {path}")
    print(f"Allocated cores: {dataset_cores}/{total_cores}")
    print(f"{'=' * 70}")
    dataset_summary_lines: List[str] = []
    dataset_summary_lines.append(f"Dataset: {ds_key}")
    dataset_summary_lines.append(f"Directed: {cfg.directed}")
    dataset_summary_lines.append(f"File: {path.name}")
    dataset_summary_lines.append(
        f"Max nodes: {args.max_nodes if args.max_nodes else 'All'}"
    )
    dataset_summary_lines.append(f"Allocated cores: {dataset_cores}/{total_cores}")
    baseline_stats: List[Dict[str, object]] = []
    baseline_graph = None
    if args.baseline == "esa":
        print(
            f"[baseline] ESA sampling before RAND-ESU (samples={args.baseline_samples}, seeds={args.seed})",
            flush=True,
        )
        baseline_graph = load_snap_graph(
            path,
            directed=cfg.directed,
            max_nodes=args.max_nodes,
            max_degree=cfg.max_degree,
        )
        if cfg.max_degree:
            print(
                f"[baseline] Applied max_degree={cfg.max_degree} filter: n={baseline_graph.number_of_nodes():,} m={baseline_graph.number_of_edges():,}",
                flush=True,
            )
        dataset_summary_lines.append("-- ESA baseline runs --")
        for k in args.k:
            for q in args.q:
                for seed in args.seed:
                    try:
                        baseline_result = run_esa_baseline(
                            baseline_graph,
                            ds_key,
                            cfg.directed,
                            k=k,
                            samples=args.baseline_samples,
                            seed=seed,
                            output_dir=args.output_dir,
                            q=q,
                            schedule=args.schedule,
                            max_retries=args.baseline_max_retries,
                            emit_plot=args.baseline_plot,
                            compute_probabilities=args.esa_probability_correction,
                        )
                        dataset_summary_lines.append(
                            "\n".join(
                                filter(
                                    None,
                                    [
                                        f"ESA baseline k={k} q={q} seed={seed}",
                                        f"  samples={baseline_result.total_samples} unique_classes={baseline_result.unique_classes}",
                                        f"  runtime_sec={baseline_result.runtime_sec:.3f}",
                                        f"  csv={baseline_result.csv_path.relative_to(args.output_dir)}",
                                        f"  meta={baseline_result.meta_path.relative_to(args.output_dir)}",
                                        (
                                            f"  plot={baseline_result.plot_path.relative_to(args.output_dir)}"
                                            if baseline_result.plot_path is not None
                                            else ""
                                        ),
                                    ],
                                )
                            ).strip()
                        )
                        baseline_stats.append(
                            {
                                "k": k,
                                "q": q,
                                "seed": seed,
                                "total_samples": baseline_result.total_samples,
                                "unique_classes": baseline_result.unique_classes,
                                "runtime_sec": baseline_result.runtime_sec,
                                "csv_path": baseline_result.csv_path,
                                "meta_path": baseline_result.meta_path,
                                "plot_path": baseline_result.plot_path,
                            }
                        )
                    except Exception as exc:
                        dataset_summary_lines.append(
                            f"ESA baseline failed for k={k} q={q} seed={seed}: {exc}"
                        )
        dataset_summary_lines.append("-- End ESA baseline --")
        baseline_graph = None

    t_ds0 = time.time()
    all_results = []

    # Process each k-value separately for optimal resource allocation
    for k in args.k:
        if k >= 6:
            print(
                f"[warn] k={k} can be very expensive; consider lowering q or using 'fine' schedule."
            )
        # Calculate total experiments for this k-value
        total_experiments = len(args.q) * len(args.seed)

        # Calculate optimal worker allocation based on experiments vs cores
        max_workers_k, cores_per_run = calculate_worker_allocation(
            max_workers, total_experiments
        )

        print(f"\n{'─' * 70}")
        print(
            f"k={k} | {total_experiments} experiments | {max_workers_k} workers × {cores_per_run} cores/run = {max_workers_k * cores_per_run}/{max_workers} cores"
        )
        print(f"{'─' * 70}")

        # Build experiments for this k value
        experiments = []
        for q in args.q:
            for seed in args.seed:
                experiments.append(
                    (
                        ds_key,
                        path,
                        cfg.directed,
                        k,
                        q,
                        args.schedule,
                        args.child_selection,
                        seed,
                        args.output_dir,
                        args.max_nodes,
                        cores_per_run,  # Pass cores_per_run for future intra-parallelism
                        cfg.max_degree,  # Pass max_degree from dataset config
                        args.memory_optimized,  # Flag for memory optimized mode
                    )
                )

        print(f"🚀 Starting {len(experiments)} runs for k={k}...", flush=True)
        t_k0 = time.time()

        # Run experiments in parallel
        completed = 0
        results = []
        with ProcessPoolExecutor(max_workers=max_workers_k) as executor:
            # Submit all tasks for this k
            future_to_exp = {
                executor.submit(run_single_experiment, exp): exp
                for exp in experiments
            }

            # Process as they complete
            for future in as_completed(future_to_exp):
                exp = future_to_exp[future]
                try:
                    result = future.result()
                    results.append(result)
                    all_results.append(result)
                    completed += 1

                    print(
                        f"  ✓ [{completed}/{len(experiments)}] k={result['k']} q={result['q']} seed={result['seed']} | "
                        f"samples={result['total_samples']} classes={result['unique_classes']} | "
                        f"time={result['runtime_sec']:.2f}s",
                        flush=True,
                    )
                except Exception as e:
                    k_val, q, seed = exp[3], exp[4], exp[7]
                    print(
                        f"  ✗ [{completed + 1}/{len(experiments)}] k={k_val} q={q} seed={seed} FAILED: {e}",
                        flush=True,
                    )

        t_k1 = time.time()
        k_elapsed = t_k1 - t_k0

        if results:
            avg_time = sum(r["runtime_sec"] for r in results) / len(results)
            print(
                f"\n✓ k={k} completed: {len(results)} runs in {k_elapsed:.1f}s (avg {avg_time:.1f}s/run)"
            )

    t_ds1 = time.time()
    ds_elapsed = t_ds1 - t_ds0

    print(f"\n{'─' * 70}")
    print(f"✓ COMPLETED DATASET: {ds_key}")
    print(
        f"  Runs completed: {len(all_results)}/{len(args.k) * len(args.q) * len(args.seed)}"
    )
    print(f"  Dataset time: {ds_elapsed:.1f}s ({ds_elapsed / 60:.1f} min)")
    total_compute = 0.0
    speedup = 0.0
    if len(all_results) > 0:
        total_compute = sum(r["runtime_sec"] for r in all_results)
        speedup = total_compute / ds_elapsed if ds_elapsed > 0 else 0
        print(f"  Compute time: {total_compute:.1f}s")
        print(
            f"  Speedup: {speedup:.1f}x (parallel efficiency: {speedup / max_workers * 100:.1f}%)"
        )
    print(f"{'─' * 70}\n")
    dataset_summary_lines.append(f"\n=== Dataset completed: {ds_key} ===")
    dataset_summary_lines.append(
        f"Environment: Python {platform.python_version()} | {platform.platform()}"
    )
    dataset_summary_lines.append(
        f"k values: {args.k} | q values: {args.q} | schedule: {args.schedule} | child: {args.child_selection}"
    )
    dataset_summary_lines.append(
        f"Total runs: {len(args.k) * len(args.q) * len(args.seed)} | Wall time: {ds_elapsed:.1f}s"
    )

    # Aggregate results for each k
    for k in args.k:
        folder = args.output_dir / ds_key / f"k{k}"
        plots_dir = folder / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Load all CSV files for this k
        t_aggr0 = time.time()
        dfs = []
        for csvf in folder.glob("*.csv"):
            try:
                tmp = pd.read_csv(csvf, dtype={"signature": str})
                dfs.append(tmp)
            except Exception:
                pass

        if dfs:
            dfa = pd.concat(dfs, ignore_index=True)

            # Top-N horizontal plot
            top_n = 12
            mean_c = (
                dfa.groupby("signature")["concentration"]
                .mean()
                .sort_values(ascending=False)
            )
            top_sigs = list(mean_c.head(top_n).index)
            pseudo_counts: Dict[str, int] = {}
            for sig in top_sigs:
                sub = dfa[dfa["signature"] == sig]
                if "total_samples" in sub.columns:
                    mean_cnt = float(
                        (sub["concentration"] * sub["total_samples"]).mean()
                    )
                else:
                    mean_cnt = float(sub["count"].mean())
                pseudo_counts[str(sig)] = int(round(mean_cnt))

            plot_motif_distribution_horizontal(
                pseudo_counts,
                total=max(1, sum(pseudo_counts.values())),
                title=f"{ds_key} k={k}: Top-{top_n} motifs (mean)",
                out_path=plots_dir / f"top{top_n}_horizontal.png",
            )

            # Boxplot
            rows = []
            for sig in top_sigs:
                sub = dfa[dfa["signature"] == sig]
                for _, r in sub.iterrows():
                    rows.append(
                        {
                            "signature": str(sig),
                            "seed": r.get("seed", 0),
                            "concentration": r["concentration"],
                        }
                    )

            plot_seed_boxplot(
                rows,
                out_path=plots_dir / f"top{top_n}_boxplot.png",
                title=f"{ds_key} k={k}: Concentration variance (Top-{top_n})",
            )

            # Triad grouping for directed k=3
            if cfg.directed and k == 3:
                dft = dfa.copy()
                dft["triad"] = (
                    dft["signature"].astype(str).apply(triad_label_from_signature)
                )
                tri_means = (
                    dft.groupby("triad")["concentration"]
                    .mean()
                    .sort_values(ascending=False)
                )
                tri_map = {t: int(round(c * 10000)) for t, c in tri_means.items()}
                plot_motif_distribution(
                    tri_map,
                    total=max(1, sum(tri_map.values())),
                    title=f"{ds_key} k=3: Triad distribution (mean)",
                    out_path=plots_dir / "triads_bar.png",
                    top_n=len(tri_map),
                )

            # Realized fraction stats (if meta files present)
            try:
                import json as _json

                rf_vals = []
                for meta_file in folder.glob("*_meta.json"):
                    with open(meta_file, "r") as fm:
                        mm = _json.load(fm)
                        rf = mm.get("realized_fraction", None)
                        if rf is not None:
                            try:
                                rf_vals.append(float(rf))
                            except Exception:
                                pass
                rf_line = ""
                if rf_vals:
                    rf_mean = sum(rf_vals) / len(rf_vals)
                    rf_min = min(rf_vals)
                    rf_max = max(rf_vals)
                    rf_line = f"realized_fraction: mean={rf_mean:.6f} min={rf_min:.6f} max={rf_max:.6f}"
                dataset_summary_lines.append(f"\n-- Aggregated summary for k={k} --")
                dataset_summary_lines.append(f"seeds={args.seed} qs={args.q}")
                if rf_line:
                    dataset_summary_lines.append(rf_line)
                dataset_summary_lines.append(
                    f"plots: {plots_dir.relative_to(args.output_dir)}/top{top_n}_horizontal.png, top{top_n}_boxplot.png"
                )
            except Exception:
                pass

            t_aggr1 = time.time()
            print(
                f"[aggregate] k={k} plots generated in {t_aggr1 - t_aggr0:.2f}s -> {plots_dir}",
                flush=True,
            )

        # ESA vs RAND-ESU overlay plots (qualitative bias visualization)
        if baseline_stats:
            try:
                # Map RAND-ESU results by (k,q,seed) to enable lookup
                rand_map = {}
                for csvf in folder.glob("q*_*_seed*.csv"):
                    try:
                        # Parse q and seed from filename q{q}_{schedule}_seed{seed}.csv
                        name = csvf.stem
                        # name like: q0.1_skewed_seed1
                        parts = name.split("_")
                        q_str = (
                            parts[0][1:]
                            if parts and parts[0].startswith("q")
                            else None
                        )
                        seed_part = [p for p in parts if p.startswith("seed")]
                        seed_val = int(seed_part[0][4:]) if seed_part else None
                        if q_str is not None and seed_val is not None:
                            rand_map[(k, float(q_str), seed_val)] = csvf
                    except Exception:
                        continue
                for b in baseline_stats:
                    if b.get("k") != k:
                        continue
                    key = (k, float(b["q"]), int(b["seed"]))
                    rand_csv = rand_map.get(key)
                    if not rand_csv:
                        continue
                    # Load ESA and RAND csvs
                    import pandas as _pd

                    esa_df = _pd.read_csv(b["csv_path"], dtype={"signature": str})
                    rand_df = _pd.read_csv(rand_csv, dtype={"signature": str})
                    esa_total = (
                        int(esa_df["total_samples"].iloc[0])
                        if not esa_df.empty
                        else 0
                    )
                    rand_total = (
                        int(rand_df["total_samples"].iloc[0])
                        if not rand_df.empty
                        else 0
                    )
                    esa_freq = {
                        str(r["signature"]): int(r["count"])
                        for _, r in esa_df.iterrows()
                    }
                    rand_freq = {
                        str(r["signature"]): int(r["count"])
                        for _, r in rand_df.iterrows()
                    }
                    ov_path = generate_overlay(
                        dataset=ds_key,
                        k=k,
                        q=float(b["q"]),
                        seed=int(b["seed"]),
                        esa_freq=esa_freq,
                        esa_total=esa_total,
                        rand_freq=rand_freq,
                        rand_total=rand_total,
                        output_dir=args.output_dir,
                    )
                    if ov_path:
                        dataset_summary_lines.append(
                            f"overlay: {ov_path.relative_to(args.output_dir)}"
                        )
            except Exception as _exc:
                dataset_summary_lines.append("[WARN] Overlay generation failed")

    sig_df = None
    direct_df = None

    if args.significance_method in ("edge-swap", "both"):
        if not cfg.directed:
            msg = "[significance] Skipping edge-swap significance for undirected dataset"
            print(msg, flush=True)
            dataset_summary_lines.append(msg)
        else:
            sig_swaps_per_edge = (
                args.significance_swaps_per_edge
                if args.significance_swaps_per_edge is not None
                else 100
            )
            print(
                f"[significance] Edge-swap ensemble for {ds_key} "
                f"(k={args.significance_k}, q={args.significance_q}, random_graphs={args.significance_random_graphs})"
            )
            try:
                sig_result = run_edge_swap_significance(
                    dataset=ds_key,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    k=args.significance_k,
                    q=args.significance_q,
                    schedule=args.schedule,
                    seeds=args.significance_seeds,
                    random_graphs=args.significance_random_graphs,
                    swaps_per_edge=sig_swaps_per_edge,
                    max_nodes=args.max_nodes,
                    use_existing_rand_esu=True,  # Load from existing CSVs for consistency
                    full_enumeration=args.significance_full_enumeration,
                    max_workers=max_workers,
                )

                dataset_summary_lines.append("\n-- Significance (edge swaps) --")
                dataset_summary_lines.append(
                    f"k={args.significance_k} q={args.significance_q} random_graphs={args.significance_random_graphs} swaps_per_edge={sig_swaps_per_edge}"
                )
                dataset_summary_lines.append(f"seeds={args.significance_seeds}")

                sig_csv_path = sig_result.get("csv_path")
                sig_meta_path = sig_result.get("meta_path")
                try:
                    sig_csv_rel = (
                        sig_csv_path.relative_to(args.output_dir)
                        if sig_csv_path
                        else None
                    )
                except Exception:
                    sig_csv_rel = sig_csv_path
                try:
                    sig_meta_rel = (
                        sig_meta_path.relative_to(args.output_dir)
                        if sig_meta_path
                        else None
                    )
                except Exception:
                    sig_meta_rel = sig_meta_path

                if sig_csv_rel:
                    dataset_summary_lines.append(f"csv: {sig_csv_rel}")
                if sig_meta_rel:
                    dataset_summary_lines.append(f"meta: {sig_meta_rel}")

                sig_df = sig_result.get("df")
                if sig_df is not None and not sig_df.empty:
                    sig_display = sig_df.copy()
                    if "signature" in sig_display.columns:
                        sig_display["signature"] = sig_display["signature"].astype(str)
                    if args.significance_k == 3 and cfg.directed:
                        sig_display["triad"] = (
                            sig_display["signature"]
                            .astype(str)
                            .apply(triad_label_from_signature)
                        )
                    top_sig = sig_display.sort_values("z_score", ascending=False).head(
                        5
                    )
                    dataset_summary_lines.append(
                        "note: enrichment_ratio=orig_concentration/rand_mean_concentration is paper-aligned; z-scores are for reference and can be extreme when rand_std is tiny."
                    )
                    dataset_summary_lines.append("top_z_scores:")
                    for _, row in top_sig.iterrows():
                        label_value = row.get("triad", row.get("signature"))
                        label = str(label_value) if label_value is not None else "-"
                        dataset_summary_lines.append(
                            f"  {label}: z={row['z_score']:.3f} ratio={row.get('enrichment_ratio', float('nan')):.3f} orig={row['orig_concentration']:.5f} rand_mean={row['rand_mean_concentration']:.5f}"
                        )

                    if "enrichment_ratio" in sig_display.columns:
                        top_ratio = sig_display.sort_values(
                            "enrichment_ratio", ascending=False
                        ).head(5)
                        dataset_summary_lines.append("top_enrichment_ratios:")
                        for _, row in top_ratio.iterrows():
                            label_value = row.get("triad", row.get("signature"))
                            label = str(label_value) if label_value is not None else "-"
                            dataset_summary_lines.append(
                                f"  {label}: ratio={row['enrichment_ratio']:.3f} orig={row['orig_concentration']:.5f} rand_mean={row['rand_mean_concentration']:.5f}"
                            )

                    if (
                        args.significance_summary
                        and args.significance_k == 3
                        and cfg.directed
                    ):
                        compare_dir = (
                            args.output_dir / ds_key / "k3" / "significance_compare"
                        )
                        compare_dir.mkdir(parents=True, exist_ok=True)
                        compare_df = sig_display.copy()
                        comp_csv = compare_dir / "significance_ensemble_baseline.csv"
                        compare_df.to_csv(comp_csv, index=False)
                        comp_txt = compare_dir / "significance_ensemble_summary.txt"
                        with open(comp_txt, "w") as f:
                            f.write(
                                f"Ensemble baseline (edge swaps) for {ds_key}, k=3, seeds={args.significance_seeds}\n"
                            )
                            f.write(
                                f"random_graphs={args.significance_random_graphs} swaps_per_edge={sig_swaps_per_edge} q={args.significance_q} schedule={args.schedule}\n"
                            )
                            f.write(
                                "Metrics: enrichment_ratio=orig_concentration/rand_mean_concentration (paper-aligned). "
                                "Z-scores are included for reference and can be extreme when rand_std_concentration is tiny.\n"
                            )
                            f.write(
                                f"Top-10 triads by mean concentration in ensemble (rand_mean_concentration):\n"
                            )
                            top_mean = compare_df.sort_values(
                                "rand_mean_concentration", ascending=False
                            ).head(10)
                            for _, r in top_mean.iterrows():
                                f.write(
                                    f"  {r.get('triad', r['signature'])}: mean={r['rand_mean_concentration']:.6g} std={r.get('rand_std_concentration', float('nan')):.6g}\n"
                                )
                            if "enrichment_ratio" in compare_df.columns:
                                top_ratio = compare_df.sort_values(
                                    "enrichment_ratio", ascending=False
                                ).head(10)
                                f.write("\nTop-10 triads by enrichment ratio (orig vs ensemble mean):\n")
                                for _, r in top_ratio.iterrows():
                                    f.write(
                                        f"  {r.get('triad', r['signature'])}: ratio={r['enrichment_ratio']:.6g} orig={r['orig_concentration']:.6g} rand_mean={r['rand_mean_concentration']:.6g}\n"
                                    )
                        try:
                            cmp_csv_rel = comp_csv.relative_to(args.output_dir)
                        except Exception:
                            cmp_csv_rel = comp_csv
                        try:
                            cmp_txt_rel = comp_txt.relative_to(args.output_dir)
                        except Exception:
                            cmp_txt_rel = comp_txt
                        dataset_summary_lines.append(
                            f"comparison_outputs: {cmp_csv_rel}, {cmp_txt_rel}"
                        )
                print("[significance] Edge-swap ensemble completed", flush=True)
            except Exception as exc:
                print(f"[significance] FAILED for {ds_key}: {exc}", flush=True)
                dataset_summary_lines.append(f"Significance failed: {exc}")
    if args.significance_method in ("direct", "both"):
        # Direct BC-inspired significance only for directed k=3, per article's demo table
        if cfg.directed and 3 in args.k:
            print(
                f"[significance] Direct (BC-inspired) triad expectations for {ds_key} (T={args.direct_T})"
            )
            try:
                from src.experiments.significance_direct_bender_canfield import (
                    run_direct_significance_bc,
                )

                dr = run_direct_significance_bc(
                    dataset=ds_key,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    T=args.direct_T,
                    seed=args.direct_seed,
                    max_nodes=args.max_nodes,
                    include_lambda_correction=args.include_lambda_correction,
                    max_workers=max_workers,
                )
                direct_df = dr.get("df")
                try:
                    csv_rel = dr["csv_path"].relative_to(args.output_dir)
                except Exception:
                    csv_rel = dr["csv_path"]
                dataset_summary_lines.append("-- Significance (direct BC-inspired) --")
                dataset_summary_lines.append(f"csv: {csv_rel}")
            except Exception as exc:
                print(
                    f"[significance] Direct-BC FAILED for {ds_key}: {exc}",
                    flush=True,
                )
                dataset_summary_lines.append(
                    f"Significance (direct BC) failed: {exc}"
                )

    # Automatic comparison if both methods produced results (for k=3 directed)
    if (
        sig_df is not None
        and direct_df is not None
        and args.significance_k == 3
        and cfg.directed
    ):
        print(
            f"[significance] Generating comparison plots (Ensemble vs Direct) for {ds_key}...",
            flush=True,
        )
        try:
            compare_dir = args.output_dir / ds_key / "k3" / "significance_compare"
            compare_dir.mkdir(parents=True, exist_ok=True)

            # Prepare direct DF for merge
            ddf = direct_df.copy()
            ddf.rename(
                columns={
                    "observed_concentration": "direct_bc_observed_concentration",
                    "expected_concentration": "direct_bc_expected_concentration",
                    "enrichment_ratio": "direct_bc_enrichment_ratio",
                    "z_score": "direct_bc_z_score",
                },
                inplace=True,
            )

            # Ensure signature is string for merge
            if "signature" in sig_df.columns:
                sig_df["signature"] = sig_df["signature"].astype(str)
            if "signature" in ddf.columns:
                ddf["signature"] = ddf["signature"].astype(str)

            merged = pd.merge(
                sig_df,
                ddf,
                how="inner",
                left_on="signature",
                right_on="signature",
            )

            merged_out = compare_dir / "significance_ensemble_vs_direct.csv"
            merged.to_csv(merged_out, index=False)

            # Scatter plots
            scatter1 = compare_dir / "scatter_rand_mean_vs_direct_expected_bc.png"
            plot_scatter_xy(
                merged,
                x_col="rand_mean_concentration",
                y_col="direct_bc_expected_concentration",
                color_col="direct_bc_z_score",
                out_path=scatter1,
                title=f"{ds_key} Triads: Ensemble mean vs Direct-BC expected",
                x_label="Ensemble mean concentration",
                y_label="Direct-BC expected concentration",
            )

            scatter2 = compare_dir / "scatter_orig_vs_direct_expected_bc.png"
            plot_scatter_xy(
                merged,
                x_col="orig_concentration",
                y_col="direct_bc_expected_concentration",
                color_col="direct_bc_z_score",
                out_path=scatter2,
                title=f"{ds_key} Triads: Original vs Direct-BC expected",
                x_label="Original concentration (RAND-ESU)",
                y_label="Direct-BC expected concentration",
            )

            dataset_summary_lines.append("-- Significance Comparison (Ensemble vs Direct) --")
            try:
                sc1_rel = scatter1.relative_to(args.output_dir)
                sc2_rel = scatter2.relative_to(args.output_dir)
                dataset_summary_lines.append(f"plots: {sc1_rel}, {sc2_rel}")
            except Exception:
                pass

            print(
                f"[significance] Comparison plots saved to {compare_dir}",
                flush=True,
            )
        except Exception as exc:
            # If the comparison failed (e.g. direct method produced no rows), we catch it here
            # and print a warning instead of crashing the whole dataset loop.
            print(
                f"[significance] Comparison generation skipped/failed: {exc}",
                flush=True,
            )

    # Write detailed per-seed summary_results.txt
    summary_results_lines: List[str] = []
    summary_results_lines.append(f"Dataset: {ds_key}")
    summary_results_lines.append(f"Directed: {cfg.directed}")
    summary_results_lines.append(f"k values: {args.k}")
    summary_results_lines.append(f"q values: {args.q}")
    summary_results_lines.append(f"Seeds: {args.seed}")
    summary_results_lines.append(f"Baseline: {args.baseline}")
    summary_results_lines.append("")
    if baseline_stats:
        summary_results_lines.append("ESA baseline results per seed:")
        for b in sorted(baseline_stats, key=lambda x: (x["k"], x["q"], x["seed"])):
            try:
                csv_rel = b["csv_path"].relative_to(args.output_dir)
            except Exception:
                csv_rel = b["csv_path"]
            try:
                meta_rel = b["meta_path"].relative_to(args.output_dir)
            except Exception:
                meta_rel = b["meta_path"]
            plot_rel = None
            if b.get("plot_path") is not None:
                try:
                    plot_rel = b["plot_path"].relative_to(args.output_dir)
                except Exception:
                    plot_rel = b["plot_path"]
            summary_results_lines.append(
                f"  k={b['k']} q={b['q']} seed={b['seed']} samples={b['total_samples']} unique={b['unique_classes']} runtime_sec={b['runtime_sec']:.3f}"
            )
            summary_results_lines.append(f"    csv={csv_rel}")
            summary_results_lines.append(f"    meta={meta_rel}")
            if plot_rel:
                summary_results_lines.append(f"    plot={plot_rel}")
        total_baseline_samples = sum(b["total_samples"] for b in baseline_stats)
        mean_baseline_runtime = sum(b["runtime_sec"] for b in baseline_stats) / len(
            baseline_stats
        )
        summary_results_lines.append(
            f"Baseline aggregate: runs={len(baseline_stats)} total_samples={total_baseline_samples} mean_runtime_sec={mean_baseline_runtime:.3f}"
        )
        summary_results_lines.append("")
    else:
        summary_results_lines.append("ESA baseline results per seed: not run")
        summary_results_lines.append("")

    summary_results_lines.append("RAND-ESU results per seed:")
    if all_results:
        for r in sorted(all_results, key=lambda x: (x["k"], x["q"], x["seed"])):
            summary_results_lines.append(
                f"  k={r['k']} q={r['q']} seed={r['seed']} samples={r['total_samples']} unique={r['unique_classes']} runtime_sec={r['runtime_sec']:.3f} load_sec={r['load_sec']:.3f}"
            )
        total_rand_samples = sum(r["total_samples"] for r in all_results)
        mean_rand_runtime = sum(r["runtime_sec"] for r in all_results) / len(
            all_results
        )
        mean_rand_unique = sum(r["unique_classes"] for r in all_results) / len(
            all_results
        )
        summary_results_lines.append(
            f"RAND-ESU aggregate: runs={len(all_results)} total_samples={total_rand_samples} mean_runtime_sec={mean_rand_runtime:.3f} mean_unique_classes={mean_rand_unique:.3f}"
        )
        summary_results_lines.append(
            f"Parallel metrics: compute_time_sec={total_compute:.3f} wall_time_sec={ds_elapsed:.3f} speedup={speedup:.3f} efficiency_percent={(speedup / max_workers * 100 if max_workers else 0):.2f}"
        )
    else:
        summary_results_lines.append("  No RAND-ESU runs recorded")

    if args.run_edge_swap_significance:
        summary_results_lines.append("")
        summary_results_lines.append("Significance runs: see summary.txt for details")

    dataset_results_path = args.output_dir / ds_key / "summary_results.txt"
    try:
        dataset_results_path.write_text("\n".join(summary_results_lines) + "\n")
    except Exception:
        pass

    ds_summary_path = args.output_dir / ds_key / "summary.txt"
    ds_summary_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ds_summary_path.write_text("\n".join(dataset_summary_lines) + "\n")
    except Exception:
        pass

    return {
        "ds_key": ds_key,
        "status": "completed",
        "dataset_time_sec": ds_elapsed,
        "runs": len(all_results),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run RAND-ESU over multiple datasets, ks, qs, and seeds in PARALLEL"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASETS.keys()),
        help="Dataset keys to run",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="*",
        default=[3, 4, 5, 6],
        help="Motif sizes (article tests k=3,4,5,6)",
    )
    parser.add_argument(
        "--q",
        type=float,
        nargs="*",
        default=[0.1],
        help="Sampling fractions (article uses 10%% = 0.1)",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="fine",
        choices=["fine", "coarse", "geometric", "skewed"],
        help="Depth prob schedule: 'fine' (1,...,1,q) recommended in article Figure 3b",
    )
    parser.add_argument(
        "--child-selection",
        type=str,
        default="bernoulli",
        choices=["bernoulli", "balanced"],
        help="Child selection strategy",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="*",
        default=[1, 2, 3],
        help="Random seeds (multiple for variance estimation)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Optional node cap for smoke tests (applies to baseline and workers)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive prompts and use defaults",
    )
    # Memory optimization flags: default ON, allow disabling with --no-memory-optimized, keep --memory-optimized for back-compat
    parser.add_argument(
        "--memory-optimized",
        dest="memory_optimized",
        action="store_true",
        help="Use memory-lean parallel counting (recommended for k>=4)",
    )
    parser.add_argument(
        "--no-memory-optimized",
        dest="memory_optimized",
        action="store_false",
        help="Disable memory-lean counting (not recommended for k>=4)",
    )
    parser.set_defaults(memory_optimized=True)
    parser.add_argument(
        "--baseline",
        type=str,
        default="esa",
        choices=["none", "esa"],
        help="Baseline algorithm to run before RAND-ESU",
    )
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=2000,
        help="Number of ESA samples per (dataset,k,q,seed) when baseline=esa",
    )
    parser.add_argument(
        "--baseline-max-retries",
        type=int,
        default=40,
        help="Max retries for ESA baseline expansion",
    )
    parser.add_argument(
        "--no-baseline-plot",
        dest="baseline_plot",
        action="store_false",
        help="Disable ESA baseline motif plots",
    )
    parser.add_argument(
        "--esa-probability-correction",
        action="store_true",
        help="Enable ESA probability correction (Equation 1). WARNING: O(k^k) per sample, very slow!",
    )
    parser.set_defaults(baseline_plot=True)
    # Significance method selection
    parser.add_argument(
        "--significance-method",
        type=str,
        default="direct",
        choices=["none", "edge-swap", "direct", "both"],
        help="Significance method: none | edge-swap (ensemble) | direct (BC-inspired) | both",
    )
    parser.add_argument(
        "--run-edge-swap-significance",
        action="store_true",
        help="[Deprecated] If provided, uses edge-swap significance; prefer --significance-method",
    )
    parser.add_argument(
        "--significance-k",
        type=int,
        default=3,
        help="Motif size for significance analysis (only k=3 triads supported; other values ignored)",
    )
    parser.add_argument(
        "--significance-q",
        type=float,
        default=None,
        help="Sampling fraction used for significance runs (default: min of RAND-ESU q values)",
    )
    parser.add_argument(
        "--significance-seeds",
        type=int,
        nargs="*",
        default=[1, 2, 3],
        help="Seeds for significance sampling on the original graph (edge-swap method)",
    )
    parser.add_argument(
        "--significance-random-graphs",
        type=int,
        default=50,
        help="Number of randomized graphs for edge-swap ensemble",
    )
    parser.add_argument(
        "--significance-swaps-per-edge",
        type=int,
        default=None,
        help="Switch factor for edge swaps (default: 100 directed, 10 undirected)",
    )
    parser.add_argument(
        "--significance-full-enumeration",
        dest="significance_full_enumeration",
        action="store_true",
        help="Use full enumeration for edge-swap random graphs (default)",
    )
    parser.add_argument(
        "--no-significance-full-enumeration",
        dest="significance_full_enumeration",
        action="store_false",
        help="Use sampling for edge-swap random graphs",
    )
    parser.set_defaults(significance_full_enumeration=True)
    parser.add_argument(
        "--direct-T",
        type=int,
        default=20000,
        help="Triples sampled for direct (BC) method",
    )
    parser.add_argument(
        "--direct-seed", type=int, default=1, help="Random seed for direct (BC) method"
    )
    parser.add_argument(
        "--include-lambda-correction",
        action="store_true",
        help="Include λ correction term in direct BC formula (more accurate)",
    )
    parser.add_argument(
        "--no-significance-summary",
        dest="significance_summary",
        action="store_false",
        help="Skip writing ensemble comparison summaries for directed k=3",
    )
    parser.set_defaults(significance_summary=True)

    args = parser.parse_args()

    # Interactive mode: prompt for selections (default behavior unless --no-interactive)
    if not args.no_interactive:
        print("[interactive] Press Enter to accept defaults in brackets.")
        print("(Defaults are based on the Wernicke 2005 paper recommendations)\n")

        # Datasets
        ds_default = ",".join(DATASETS.keys())
        ds_in = input(f"Datasets (comma-separated or 'all') [{ds_default}]: ").strip()
        if ds_in and ds_in.lower() != "all":
            selected = []
            # Create a mapping for case-insensitive lookup
            ds_map = {k.lower(): k for k in DATASETS.keys()}

            for s in ds_in.split(","):
                s_clean = s.strip()
                if s_clean in DATASETS:
                    selected.append(s_clean)
                elif s_clean.lower() in ds_map:
                    selected.append(ds_map[s_clean.lower()])
                else:
                    print(f"[warning] Dataset '{s_clean}' not found. Skipping.")
            args.datasets = selected

        # k values
        k_str = ",".join(map(str, args.k))
        k_in = input(f"k sizes (comma-separated) [{k_str}]: ").strip()
        if k_in:
            args.k = [int(s.strip()) for s in k_in.split(",") if s.strip()]

        # q values
        q_str = ",".join(map(str, args.q))
        q_in = input(f"q values (sampling fraction, 0.1=10%) [{q_str}]: ").strip()
        if q_in:
            args.q = [float(s.strip()) for s in q_in.split(",") if s.strip()]

        # schedule
        sch_in = input(
            f"schedule (fine|coarse|geometric|skewed - 'fine' recommended per article) [{args.schedule}]: "
        ).strip()
        if sch_in in ["fine", "coarse", "geometric", "skewed"]:
            args.schedule = sch_in

        # child-selection
        ch_in = input(
            f"child-selection (bernoulli|balanced) [{args.child_selection}]: "
        ).strip()
        if ch_in in ["bernoulli", "balanced"]:
            args.child_selection = ch_in

        # seeds
        sd_str = ",".join(map(str, args.seed))
        sd_in = input(f"seeds (multiple for variance estimation) [{sd_str}]: ").strip()
        if sd_in:
            args.seed = [int(s.strip()) for s in sd_in.split(",") if s.strip()]

        # max-workers
        default_workers = cpu_count()
        mw_in = input(f"max-workers (parallel processes) [{default_workers}]: ").strip()
        if mw_in:
            args.max_workers = int(mw_in)

        mn_in = input(
            f"max-nodes (None for all nodes, or integer for testing) [{args.max_nodes}]: "
        ).strip()
        if mn_in:
            args.max_nodes = None if mn_in.lower() in ("none", "") else int(mn_in)

        # memory-optimized counting prompt (default: enabled)
        mo_default = "y" if args.memory_optimized else "n"
        mo_in = (
            input(
                f"memory-optimized counting (recommended for k>=4)? (y/n) [{mo_default}]: "
            )
            .strip()
            .lower()
        )
        if mo_in in ("y", "n"):
            args.memory_optimized = mo_in == "y"

        # paths
        dd_in = input(f"data-dir [{args.data_dir}]: ").strip()
        if dd_in:
            args.data_dir = Path(dd_in)
        od_in = input(f"output-dir [{args.output_dir}]: ").strip()
        if od_in:
            args.output_dir = Path(od_in)

        base_in = input(f"baseline (none|esa) [{args.baseline}]: ").strip()
        if base_in in ("none", "esa"):
            args.baseline = base_in
        if args.baseline == "esa":
            bs_in = input(
                f"baseline samples per run [{args.baseline_samples}]: "
            ).strip()
            if bs_in:
                args.baseline_samples = int(bs_in)
            br_in = input(
                f"baseline max retries [{args.baseline_max_retries}]: "
            ).strip()
            if br_in:
                args.baseline_max_retries = int(br_in)
            bp_in = (
                input(f"baseline plots? (y/n) [{'y' if args.baseline_plot else 'n'}]: ")
                .strip()
                .lower()
            )
            if bp_in in ("y", "n"):
                args.baseline_plot = bp_in == "y"

        # Significance method interactive selection
        if 3 in args.k:
            sm_in = input(
                f"significance method (none|edge-swap|direct|both) [{args.significance_method}]: "
            ).strip()
            if sm_in in ("none", "edge-swap", "direct", "both"):
                args.significance_method = sm_in

            # Back-compat: Only ask if method is 'none' to allow enabling via old prompt style
            if args.significance_method == "none":
                sig_prompt_default = "y" if args.run_edge_swap_significance else "n"
                es_yn = (
                    input(f"run edge-swap significance? (y/n) [{sig_prompt_default}]: ")
                    .strip()
                    .lower()
                )
                if es_yn == "y":
                    args.significance_method = "edge-swap"

            if args.significance_method in ("edge-swap", "both"):
                # Force k=3 for significance if we are in this block (since we only enter if 3 in args.k)
                # But user might want to change it? The user said "edge swap is only run for k=3".
                # So we skip asking for k and force it to 3.
                args.significance_k = 3

                sig_q_default = (
                    args.significance_q
                    if args.significance_q is not None
                    else (min(args.q) if args.q else 0.1)
                )
                sq_in = input(
                    f"significance sampling fraction q [{sig_q_default}]: "
                ).strip()
                if sq_in:
                    args.significance_q = float(sq_in)

                ss_default = ",".join(map(str, args.significance_seeds))
                ss_in = input(
                    f"significance seeds (comma-separated) [{ss_default}]: "
                ).strip()
                if ss_in:
                    args.significance_seeds = [
                        int(s.strip()) for s in ss_in.split(",") if s.strip()
                    ]

                sr_in = input(
                    f"significance randomized graphs [{args.significance_random_graphs}]: "
                ).strip()
                if sr_in:
                    args.significance_random_graphs = int(sr_in)

                sp_default = (
                    args.significance_swaps_per_edge
                    if args.significance_swaps_per_edge is not None
                    else "auto"
                )
                sp_in = input(
                    f"significance swaps per edge multiplier [{sp_default}]: "
                ).strip()
                if sp_in:
                    args.significance_swaps_per_edge = int(sp_in)

                sfe_default = "y" if args.significance_full_enumeration else "n"
                sfe_in = (
                    input(
                        f"edge-swap full enumeration for random graphs? (y/n) [{sfe_default}]: "
                    )
                    .strip()
                    .lower()
                )
                if sfe_in in ("y", "n"):
                    args.significance_full_enumeration = sfe_in == "y"

                ssumm_in = (
                    input(
                        f"write significance summary outputs? (y/n) [{'y' if args.significance_summary else 'n'}]: "
                    )
                    .strip()
                    .lower()
                )
                if ssumm_in in ("y", "n"):
                    args.significance_summary = ssumm_in == "y"
            elif args.significance_method == "direct":
                dt_in = input(f"direct-BC triples T [{args.direct_T}]: ").strip()
                if dt_in:
                    args.direct_T = int(dt_in)
                ds_in = input(f"direct-BC seed [{args.direct_seed}]: ").strip()
                if ds_in:
                    args.direct_seed = int(ds_in)
        else:
            args.significance_method = "none"
            print(
                "[interactive] Skipping significance selection (only available for k=3)"
            )

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Enforce significance only for k=3 (CLI safety check)
    if args.significance_method != "none" and 3 not in args.k:
        print("\n[info] Significance analysis is only enabled when k=3 is selected.")
        print("[info] Disabling significance method.")
        args.significance_method = "none"

    # Determine significance defaults
    if (
        args.significance_method in ("edge-swap", "both")
    ) and args.significance_q is None:
        args.significance_q = min(args.q) if args.q else 0.1
    if args.significance_seeds is None:
        args.significance_seeds = [1, 2, 3]
    else:
        args.significance_seeds = list(args.significance_seeds)

    # Guard: Significance analysis only supports k=3
    if args.significance_method != "none" and args.significance_k != 3:
        print(
            f"[WARNING] Significance analysis only supports k=3 (triads). "
            f"Overriding --significance-k={args.significance_k} to k=3."
        )
        args.significance_k = 3

    # Determine number of workers (total core budget)
    max_workers = args.max_workers if args.max_workers else cpu_count()

    # Validate datasets and compute per-dataset core allocations
    datasets_to_run = [ds for ds in args.datasets if ds in DATASETS]
    invalid_datasets = [ds for ds in args.datasets if ds not in DATASETS]
    if invalid_datasets:
        print(
            f"[WARN] Unknown dataset keys (skipping): {', '.join(invalid_datasets)}",
            flush=True,
        )
    if not datasets_to_run:
        print("[ERROR] No valid datasets selected. Exiting.", flush=True)
        return

    dataset_workers = min(len(datasets_to_run), max_workers)
    dataset_workers = max(1, dataset_workers)
    cores_per_dataset_base = max(1, max_workers // dataset_workers)
    extra_cores = max_workers % dataset_workers
    dataset_core_allocations: Dict[str, int] = {}
    for i, ds_key in enumerate(datasets_to_run):
        bonus = 1 if (i % dataset_workers) < extra_cores else 0
        dataset_core_allocations[ds_key] = cores_per_dataset_base + bonus

    # Print initial configuration summary
    print("\n" + "=" * 70)
    print("RAND-ESU PARALLEL EXPERIMENT CONFIGURATION")
    print("=" * 70)
    print(
        f"Datasets:        {', '.join(datasets_to_run)} ({len(datasets_to_run)} dataset(s))"
    )
    print(f"k values:        {args.k}")
    print(f"q values:        {args.q}")
    print(f"Schedule:        {args.schedule}")
    print(f"Child selection: {args.child_selection}")
    print(f"Memory optimized: {args.memory_optimized}")
    print(f"Seeds:           {args.seed}")
    print(f"Max nodes:       {args.max_nodes if args.max_nodes else 'All'}")
    print(
        f"Baseline:        {args.baseline} (samples={args.baseline_samples if args.baseline == 'esa' else 'n/a'})"
    )
    if args.significance_method in ("edge-swap", "both"):
        sig_swaps_label = (
            args.significance_swaps_per_edge
            if args.significance_swaps_per_edge is not None
            else "auto"
        )
        sig_full_enum = "full_enumeration" if args.significance_full_enumeration else "sampling"
        print(
            "Edge-swap significance: enabled "
            f"(k={args.significance_k}, q={args.significance_q}, seeds={args.significance_seeds}, "
            f"random_graphs={args.significance_random_graphs}, swaps_per_edge={sig_swaps_label}, "
            f"mode={sig_full_enum})"
        )
    else:
        print("Edge-swap significance: disabled")
    print(f"Output dir:      {args.output_dir}")
    print(f"\nTotal core budget: {max_workers} (CPU cores: {cpu_count()})")
    print(
        f"Dataset parallelism: {dataset_workers} worker(s), cores/dataset: {cores_per_dataset_base} (+1 for {extra_cores} dataset(s))"
    )
    print(
        f"Total runs:       {len(datasets_to_run) * len(args.k) * len(args.q) * len(args.seed)}"
    )
    print("=" * 70 + "\n")

    # Save unified run configuration to output directory
    run_config = {
        "experiment_type": "RAND-ESU parallel with ESA baseline",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": datasets_to_run,
        "k_values": args.k,
        "q_values": args.q,
        "schedule": args.schedule,
        "child_selection": args.child_selection,
        "memory_optimized": args.memory_optimized,
        "seeds": args.seed,
        "max_nodes": args.max_nodes,
        "max_workers": max_workers,
        "dataset_workers": dataset_workers,
        "cores_per_dataset_base": cores_per_dataset_base,
        "extra_dataset_cores": extra_cores,
        "dataset_core_allocations": dataset_core_allocations,
        "baseline": {
            "algorithm": args.baseline,
            "samples": args.baseline_samples if args.baseline == "esa" else None,
            "max_retries": args.baseline_max_retries
            if args.baseline == "esa"
            else None,
            "plot_enabled": args.baseline_plot if args.baseline == "esa" else None,
            "probability_correction": args.esa_probability_correction
            if args.baseline == "esa"
            else None,
        },
        "significance": {
            "method": args.significance_method,
            "k": args.significance_k if args.significance_method != "none" else None,
            "q": args.significance_q
            if args.significance_method in ("edge-swap", "both")
            else None,
            "seeds": args.significance_seeds
            if args.significance_method in ("edge-swap", "both")
            else None,
            "random_graphs": args.significance_random_graphs
            if args.significance_method in ("edge-swap", "both")
            else None,
            "swaps_per_edge": (
                args.significance_swaps_per_edge
                if args.significance_swaps_per_edge is not None
                else "auto"
            )
            if args.significance_method in ("edge-swap", "both")
            else None,
            "full_enumeration": args.significance_full_enumeration
            if args.significance_method in ("edge-swap", "both")
            else None,
            "direct_T": args.direct_T
            if args.significance_method in ("direct", "both")
            else None,
            "direct_seed": args.direct_seed
            if args.significance_method in ("direct", "both")
            else None,
            "include_lambda_correction": args.include_lambda_correction
            if args.significance_method in ("direct", "both")
            else None,
        },

        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": cpu_count(),
        },
    }
    run_config_path = args.output_dir / "run_config.json"
    with open(run_config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"[config] Saved run configuration to {run_config_path}")

    t_all0 = time.time()

    if dataset_workers > 1:
        print(
            f"[info] Running datasets in parallel: workers={dataset_workers}",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=dataset_workers) as executor:
            future_to_ds = {
                executor.submit(
                    run_dataset_pipeline,
                    args,
                    ds_key,
                    dataset_core_allocations[ds_key],
                    max_workers,
                ): ds_key
                for ds_key in datasets_to_run
            }

            for future in as_completed(future_to_ds):
                ds_key = future_to_ds[future]
                try:
                    result = future.result()
                    if result.get("status") == "skipped":
                        print(f"[WARN] Skipped dataset {ds_key}", flush=True)
                except Exception as exc:
                    print(f"[ERROR] Dataset {ds_key} failed: {exc}", flush=True)
    else:
        for ds_key in datasets_to_run:
            try:
                run_dataset_pipeline(
                    args,
                    ds_key,
                    dataset_core_allocations[ds_key],
                    max_workers,
                )
            except Exception as exc:
                print(f"[ERROR] Dataset {ds_key} failed: {exc}", flush=True)

    # Overall completion summary
    t_all_end = time.time()
    total_elapsed = t_all_end - t_all0
    total_runs = len(datasets_to_run) * len(args.k) * len(args.q) * len(args.seed)

    print("\n" + "=" * 70)
    print("✓ ALL PARALLEL EXPERIMENTS COMPLETED")
    print("=" * 70)
    print(f"Total datasets:    {len(datasets_to_run)}")
    print(f"Total runs:        {total_runs}")
    print(f"Total core budget: {max_workers}")
    print(f"Dataset workers:   {dataset_workers}")
    print(f"Wall clock time:   {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"Output dir:        {args.output_dir}")
    print("=" * 70 + "\n")

    # CONTRIBUTION: Performance scaling analysis across datasets
    print("\n" + "=" * 70)
    print("CONTRIBUTION: Performance Scaling Analysis")
    print("=" * 70)
    print("Analyzing how runtime scales with network size (n, m)...")

    scaling_data = []
    for ds_key in datasets_to_run:
        cfg = DATASETS.get(ds_key)
        if not cfg:
            continue
        path = resolve_data_path(args.data_dir, ds_key)
        G_temp = load_snap_graph(path, directed=cfg.directed, max_nodes=args.max_nodes)
        n = G_temp.number_of_nodes()
        m = G_temp.number_of_edges()

        # Aggregate runtime from all runs for this dataset
        for k in args.k:
            subdir = args.output_dir / ds_key / f"k{k}"
            if not subdir.exists():
                continue

            runtimes = []
            total_samples_list = []
            for meta_file in subdir.glob("*_meta.json"):
                try:
                    with open(meta_file, "r") as f:
                        meta = json.load(f)
                        if "runtime_sec" in meta and "total_samples" in meta:
                            runtimes.append(meta["runtime_sec"])
                            total_samples_list.append(meta["total_samples"])
                except:
                    continue

            if runtimes:
                avg_runtime = sum(runtimes) / len(runtimes)
                avg_samples = sum(total_samples_list) / len(total_samples_list)
                samples_per_sec = avg_samples / avg_runtime if avg_runtime > 0 else 0

                scaling_data.append(
                    {
                        "dataset": ds_key,
                        "k": k,
                        "n": n,
                        "m": m,
                        "avg_runtime_sec": avg_runtime,
                        "avg_samples": avg_samples,
                        "samples_per_sec": samples_per_sec,
                        "runtime_per_edge": avg_runtime / m if m > 0 else 0,
                    }
                )

    if scaling_data:
        scaling_df = pd.DataFrame(scaling_data)
        scaling_csv = args.output_dir / "performance_scaling_parallel.csv"
        scaling_df.to_csv(scaling_csv, index=False)
        print(f"✓ Saved scaling analysis: {scaling_csv}")

        print("\nPerformance Scaling Summary (Parallel Execution):")
        print("-" * 70)
        for k in sorted(scaling_df["k"].unique()):
            k_data = scaling_df[scaling_df["k"] == k].sort_values("n")
            print(f"\nk={k}:")
            print(
                f"{'Dataset':<20} {'Nodes':>10} {'Edges':>10} {'Runtime(s)':>12} {'Samples/s':>12}"
            )
            print("-" * 70)
            for _, row in k_data.iterrows():
                print(
                    f"{row['dataset']:<20} {row['n']:>10} {row['m']:>10} {row['avg_runtime_sec']:>12.2f} {row['samples_per_sec']:>12.1f}"
                )

        # Calculate scaling trend
        print("\nScaling Trends (larger networks take proportionally):")
        for k in sorted(scaling_df["k"].unique()):
            k_data = scaling_df[scaling_df["k"] == k].sort_values("n")
            if len(k_data) > 1:
                # Compare smallest to largest
                smallest = k_data.iloc[0]
                largest = k_data.iloc[-1]
                n_ratio = largest["n"] / smallest["n"]
                runtime_ratio = largest["avg_runtime_sec"] / smallest["avg_runtime_sec"]
                print(f"  k={k}: {n_ratio:.1f}x nodes → {runtime_ratio:.1f}x runtime")

        print(
            "\nNote: These are per-run runtimes. Parallel speedup reported per-dataset above."
        )
    else:
        print("[WARN] No scaling data available")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
