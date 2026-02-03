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

import pandas as pd

from src.config import DATASETS, resolve_data_path
from src.utils.io import load_snap_graph
from src.utils.motifs import count_motif_signatures, triad_label_from_signature
import platform
from src.algorithms.rand_esu import RandESUParams, rand_esu_sample, esu_enumerate, approximate_realized_fraction
from src.experiments.baselines import run_esa_baseline
from src.experiments.common import build_p_schedule
from src.utils.visualize import (
    plot_motif_distribution,
    plot_motif_distribution_horizontal,
    plot_seed_boxplot,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run RAND-ESU over multiple datasets, ks, qs, and seeds in sequence"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--datasets", nargs="*", default=list(DATASETS.keys()), help="Dataset keys to run"
    )
    parser.add_argument("--k", type=int, nargs="*", default=[3, 4, 5, 6], help="Motif sizes (article tests k=3,4,5,6)")
    parser.add_argument(
        "--q", type=float, nargs="*", default=[0.1], help="Sampling fractions (article uses 10% = 0.1)"
    )
    parser.add_argument(
        "--schedule", type=str, default="skewed", choices=["geometric", "skewed"], help="Depth prob schedule (skewed recommended in article)"
    )
    parser.add_argument(
        "--child-selection", type=str, default="bernoulli", choices=["bernoulli", "balanced"], help="Child selection strategy"
    )
    parser.add_argument("--seed", type=int, nargs="*", default=[1, 2, 3], help="Random seeds (multiple for variance estimation)")
    parser.add_argument("--max-nodes", type=int, default=None, help="Optional node cap for smoke tests")
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive prompts and use defaults")
    parser.add_argument("--baseline", type=str, default="esa", choices=["none", "esa"], help="Baseline algorithm to run before RAND-ESU runs")
    parser.add_argument("--baseline-samples", type=int, default=2000, help="Number of ESA samples per (dataset,k,q,seed) when baseline=esa")
    parser.add_argument("--baseline-max-retries", type=int, default=40, help="Max retries for ESA baseline expansion")
    parser.add_argument("--no-baseline-plot", dest="baseline_plot", action="store_false", help="Disable ESA baseline motif plots")
    parser.set_defaults(baseline_plot=True)

    args = parser.parse_args()

    # Interactive mode: prompt for selections (default behavior unless --no-interactive)
    if not args.no_interactive:
        print("[interactive] Press Enter to accept defaults in brackets.")
        print("(Defaults are based on the Wernicke 2005 paper recommendations)\n")
        
        # Datasets
        ds_default = ",".join(DATASETS.keys())
        ds_in = input(f"Datasets (comma-separated or 'all') [{ds_default}]: ").strip()
        if ds_in and ds_in.lower() != "all":
            args.datasets = [s.strip() for s in ds_in.split(",") if s.strip() in DATASETS]
        
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
        sch_in = input(f"schedule (geometric|skewed - skewed reduces variance) [{args.schedule}]: ").strip()
        if sch_in in ["geometric", "skewed"]:
            args.schedule = sch_in
        
        # child-selection
        ch_in = input(f"child-selection (bernoulli|balanced) [{args.child_selection}]: ").strip()
        if ch_in in ["bernoulli", "balanced"]:
            args.child_selection = ch_in
        
        # seeds
        sd_str = ",".join(map(str, args.seed))
        sd_in = input(f"seeds (multiple for variance estimation) [{sd_str}]: ").strip()
        if sd_in:
            args.seed = [int(s.strip()) for s in sd_in.split(",") if s.strip()]
        
        # max-nodes
        mn_in = input(f"max-nodes (None for all nodes, or integer for testing) [{args.max_nodes}]: ").strip()
        if mn_in:
            args.max_nodes = None if mn_in.lower() in ("none", "") else int(mn_in)
        
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
            bs_in = input(f"baseline samples per run [{args.baseline_samples}]: ").strip()
            if bs_in:
                args.baseline_samples = int(bs_in)
            br_in = input(f"baseline max retries [{args.baseline_max_retries}]: ").strip()
            if br_in:
                args.baseline_max_retries = int(br_in)
            bp_in = input(f"baseline plots? (y/n) [{'y' if args.baseline_plot else 'n'}]: ").strip().lower()
            if bp_in in ("y", "n"):
                args.baseline_plot = (bp_in == "y")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Progress accounting (ETA across runs only)
    total_runs = len(args.datasets) * len(args.k) * len(args.q) * len(args.seed)
    completed_runs = 0
    t_all0 = time.time()
    
    # Print initial configuration summary
    print("\n" + "="*70)
    print("RAND-ESU EXPERIMENT CONFIGURATION")
    print("="*70)
    print(f"Datasets:        {', '.join(args.datasets)} ({len(args.datasets)} dataset(s))")
    print(f"k values:        {args.k}")
    print(f"q values:        {args.q}")
    print(f"Schedule:        {args.schedule}")
    print(f"Child selection: {args.child_selection}")
    print(f"Seeds:           {args.seed}")
    print(f"Max nodes:       {args.max_nodes if args.max_nodes else 'All'}")
    print(f"Baseline:        {args.baseline} (samples={args.baseline_samples if args.baseline == 'esa' else 'n/a'})")
    print(f"Output dir:      {args.output_dir}")
    print(f"\nTotal runs:      {total_runs} (datasets × k × q × seeds)")
    print(f"Estimated time:  Will be calculated after first run")
    print("="*70 + "\n")
    
    for ds_key in args.datasets:
        cfg = DATASETS.get(ds_key)
        if cfg is None:
            print(f"[WARN] Unknown dataset key '{ds_key}', skipping")
            continue
        path = resolve_data_path(args.data_dir, ds_key)
        print(f"\n=== Dataset: {ds_key} ({'directed' if cfg.directed else 'undirected'}) ===", flush=True)
        print(f"Loading from {path} (max_nodes={args.max_nodes})...", flush=True)
        t0 = time.time()
        G = load_snap_graph(path, directed=cfg.directed, max_nodes=args.max_nodes)
        t1 = time.time()

        # Prepare summary lines per dataset
        summary_lines: List[str] = []
        summary_lines.append(f"Dataset: {ds_key}\nDirected: {cfg.directed}\nFile: {path.name}\n")
        summary_lines.append(
            "Environment:\n"
            f"  Python: {platform.python_version()}\n"
            f"  Platform: {platform.platform()}\n"
        )
        print(f"[phase.load] n={G.number_of_nodes()} m={G.number_of_edges()} took {t1 - t0:.2f}s", flush=True)

        if args.baseline == "esa":
            print(
                f"[baseline] ESA sampling before RAND-ESU (samples={args.baseline_samples}, seeds={args.seed})",
                flush=True,
            )
            summary_lines.append("-- ESA baseline runs --")
            for k in args.k:
                for q in args.q:
                    for seed in args.seed:
                        try:
                            baseline_result = run_esa_baseline(
                                G,
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
                            )
                            summary_lines.append(
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
                        except Exception as exc:
                            summary_lines.append(
                                f"ESA baseline failed for k={k} q={q} seed={seed}: {exc}"
                            )
            summary_lines.append("-- End ESA baseline --")

        for k in args.k:
            for q in args.q:
                p_depth = build_p_schedule(k, q, args.schedule)
                if k >= 6:
                    print(f"[warn] k={k} can be very expensive; consider lowering q or using 'fine' schedule.")
                for seed in args.seed:
                    print(f"[run.start] k={k} q={q} schedule={args.schedule} child={args.child_selection} seed={seed}", flush=True)
                    import random

                    random.seed(seed)
                    params = RandESUParams(k=k, p_depth=p_depth, child_selection=args.child_selection)

                    t2 = time.time()
                    samples_iter = rand_esu_sample(G, params)
                    total, freq = count_motif_signatures(G, samples_iter)
                    t3 = time.time()
                    print(f"  [phase.sample+count] samples={total} unique={len(freq)} took {t3 - t2:.2f}s", flush=True)

                    # Save results
                    records = []
                    for sig, count in freq.items():
                        records.append(
                            {
                                "dataset": ds_key,
                                "directed": cfg.directed,
                                "k": k,
                                "q": q,
                                "schedule": args.schedule,
                                "seed": seed,
                                "total_samples": total,
                                "signature": sig,
                                "count": count,
                                "concentration": (count / total) if total > 0 else 0.0,
                            }
                        )
                    df = pd.DataFrame.from_records(records)
                    subdir = args.output_dir / ds_key / f"k{k}"
                    subdir.mkdir(parents=True, exist_ok=True)
                    csv_path = subdir / f"q{q}_{args.schedule}_seed{seed}.csv"
                    t_save0 = time.time()
                    df.to_csv(csv_path, index=False)
                    t_csv = time.time()

                    try:
                        rel_file = str(path.resolve().relative_to(args.data_dir.resolve()))
                    except Exception:
                        rel_file = path.name

                    # Optional realized sampling fraction (enumeration on small graphs only)
                    realized_fraction = float('nan')
                    try:
                        if args.max_nodes and G.number_of_nodes() <= args.max_nodes:
                            all_k = sum(1 for _ in esu_enumerate(G, k))
                            rf, _ = approximate_realized_fraction(k, p_depth, total, all_k)
                            realized_fraction = rf
                    except Exception:
                        pass

                    meta = {
                        "dataset": ds_key,
                        "file": rel_file,
                        "directed": cfg.directed,
                        "n": G.number_of_nodes(),
                        "m": G.number_of_edges(),
                        "k": k,
                        "q": q,
                        "p_depth": p_depth,
                        "child_selection": params.child_selection,
                        "schedule": args.schedule,
                        "seed": seed,
                        "runtime_sec": t3 - t2,
                        "loaded_sec": t1 - t0,
                        "total_samples": total,
                        "unique_motif_classes": len(freq),
                        "realized_fraction": realized_fraction,
                        "max_nodes": args.max_nodes,
                    }
                    json_path = subdir / f"q{q}_{args.schedule}_seed{seed}_meta.json"
                    with open(json_path, "w") as f:
                        json.dump(meta, f, indent=2)
                    t_json = time.time()
                    
                    # Plot motif distribution (per-run)
                    plot_path = subdir / f"q{q}_{args.schedule}_seed{seed}_motifs.png"
                    plot_title = f"{ds_key} k={k} q={q} seed={seed} sel={params.child_selection}"
                    plot_motif_distribution(freq, total, plot_title, plot_path)
                    t_save1 = time.time()
                    
                    print(
                        f"  [phase.save] csv={t_csv-t_save0:.2f}s json={t_json-t_csv:.2f}s plot={t_save1-t_json:.2f}s total={t_save1-t_save0:.2f}s",
                        flush=True
                    )

                    # Append to summary (include realized fraction)
                    summary_lines.append(
                        "\n".join(
                            [
                                f"k={k} q={q} seed={seed}",
                                f"  samples={total} unique_classes={len(freq)}",
                                f"  schedule={args.schedule} child_selection={params.child_selection}",
                                f"  runtime_sec={t3 - t2:.3f}",
                                f"  realized_fraction={realized_fraction}",
                                f"  csv={csv_path.relative_to(args.output_dir)}",
                                f"  meta={json_path.relative_to(args.output_dir)}",
                                f"  plot={plot_path.relative_to(args.output_dir)}",
                            ]
                        )
                    )
                    # Progress ETA
                    completed_runs += 1
                    elapsed_all = time.time() - t_all0
                    avg_per_run = elapsed_all / completed_runs if completed_runs else 0.0
                    remaining = max(0, total_runs - completed_runs)
                    eta_sec = remaining * avg_per_run
                    print(
                        f"[run.done] {completed_runs}/{total_runs} runs finished | last_run={(t3 - t2)+(t_save1 - t_save0):.2f}s | ETA ~ {eta_sec/60:.1f} min",
                        flush=True,
                    )
            # After completing all q x seeds for this k, produce aggregated plots
            folder = args.output_dir / ds_key / f"k{k}"
            plots_dir = folder / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Load all runs for this k
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
                # Horizontal Top-N (mean concentration)
                top_n = 12
                mean_c = dfa.groupby("signature")["concentration"].mean().sort_values(ascending=False)
                top_sigs = list(mean_c.head(top_n).index)
                pseudo_counts: Dict[str, int] = {}
                for sig in top_sigs:
                    sub = dfa[dfa["signature"] == sig]
                    if "total_samples" in sub.columns:
                        mean_cnt = float((sub["concentration"] * sub["total_samples"]).mean())
                    else:
                        mean_cnt = float(sub["count"].mean())
                    pseudo_counts[str(sig)] = int(round(mean_cnt))
                plot_motif_distribution_horizontal(
                    pseudo_counts,
                    total=max(1, sum(pseudo_counts.values())),
                    title=f"{ds_key} k={k}: Top-{top_n} motifs (mean)",
                    out_path=plots_dir / f"top{top_n}_horizontal.png",
                )

                # Boxplot of concentration variance across seeds for top signatures
                rows = []
                for sig in top_sigs:
                    sub = dfa[dfa["signature"] == sig]
                    for _, r in sub.iterrows():
                        rows.append({"signature": str(sig), "seed": r.get("seed", 0), "concentration": r["concentration"]})
                plot_seed_boxplot(
                    rows,
                    out_path=plots_dir / f"top{top_n}_boxplot.png",
                    title=f"{ds_key} k={k}: Concentration variance (Top-{top_n})",
                )

                # Triad grouping for directed k=3
                if cfg.directed and k == 3:
                    dft = dfa.copy()
                    dft["triad"] = dft["signature"].astype(str).apply(triad_label_from_signature)
                    tri_means = dft.groupby("triad")["concentration"].mean().sort_values(ascending=False)
                    tri_map = {t: int(round(c * 10000)) for t, c in tri_means.items()}
                    plot_motif_distribution(
                        tri_map,
                        total=max(1, sum(tri_map.values())),
                        title=f"{ds_key} k=3: Triad distribution (mean)",
                        out_path=plots_dir / "triads_bar.png",
                        top_n=len(tri_map),
                    )
                t_aggr1 = time.time()
                print(f"[phase.aggregate] k={k} plots+stats took {t_aggr1 - t_aggr0:.2f}s -> {plots_dir}", flush=True)

                # Aggregated textual summary for this k
                try:
                    # Ensure correct dtypes
                    dfa["signature"] = dfa["signature"].astype(str)
                    # Distinct runs and basic aggregates
                    runs_df = (
                        dfa.groupby(["q", "seed"])  # one row per run
                        .agg(total_samples=("total_samples", "first"))
                        .reset_index()
                        .sort_values(["q", "seed"]) 
                    )
                    seeds_list = sorted(dfa["seed"].dropna().unique().tolist())
                    q_list = sorted(dfa["q"].dropna().unique().tolist())
                    total_samples_sum = int(runs_df["total_samples"].sum())
                    total_samples_mean = float(runs_df["total_samples"].mean()) if len(runs_df) else 0.0
                    total_samples_std = float(runs_df["total_samples"].std(ddof=1)) if len(runs_df) > 1 else 0.0

                    # Per-signature stats
                    sig_stats = (
                        dfa.groupby("signature").agg(
                            mean_conc=("concentration", "mean"),
                            std_conc=("concentration", "std"),
                            min_conc=("concentration", "min"),
                            max_conc=("concentration", "max"),
                            mean_count=("count", "mean"),
                            runs=("concentration", "count"),
                        )
                    )
                    sig_stats["std_conc"].fillna(0.0, inplace=True)
                    sig_stats["cv_conc"] = sig_stats.apply(
                        lambda r: (r["std_conc"] / r["mean_conc"]) if r["mean_conc"] > 0 else 0.0, axis=1
                    )
                    sig_stats_sorted = sig_stats.sort_values("mean_conc", ascending=False)

                    # Consolidated variance summary
                    conc_std_mean = float(sig_stats["std_conc"].mean() if len(sig_stats) else 0.0)
                    conc_cv_mean = float(sig_stats["cv_conc"].mean() if len(sig_stats) else 0.0)

                    # Save full per-signature stats CSV
                    sig_csv = folder / "aggregated_signature_stats.csv"
                    sig_stats_sorted.to_csv(sig_csv)

                    # Build top-12 lines for summary
                    top_lines = []
                    for sig, row in sig_stats_sorted.head(top_n).iterrows():
                        tri = (
                            triad_label_from_signature(str(sig))
                            if (cfg.directed and k == 3)
                            else "-"
                        )
                        top_lines.append(
                            f"  sig={sig} triad={tri} mean_c={row['mean_conc']:.6f} std={row['std_conc']:.6f} cv={row['cv_conc']:.3f} mean_count={row['mean_count']:.2f} runs={int(row['runs'])}"
                        )

                    # Triad-level aggregates for directed triads
                    tri_csv_path = None
                    tri_section_lines: List[str] = []
                    if cfg.directed and k == 3:
                        tri_stats = (
                            dfa.assign(triad=dfa["signature"].astype(str).apply(triad_label_from_signature))
                            .groupby("triad")
                            .agg(
                                mean_conc=("concentration", "mean"),
                                std_conc=("concentration", "std"),
                                runs=("concentration", "count"),
                            )
                            .sort_values("mean_conc", ascending=False)
                        )
                        tri_stats["std_conc"].fillna(0.0, inplace=True)
                        tri_csv_path = folder / "aggregated_triad_stats.csv"
                        tri_stats.to_csv(tri_csv_path)
                        for tri, r in tri_stats.iterrows():
                            tri_section_lines.append(
                                f"  triad={tri} mean_c={r['mean_conc']:.6f} std={r['std_conc']:.6f} runs={int(r['runs'])}"
                            )

                    # Append aggregated section to summary
                    summary_lines.append(
                        "\n".join(
                            [
                                f"-- Aggregated summary for k={k} --",
                                f"runs={len(runs_df)} seeds={seeds_list} qs={q_list}",
                                f"total_samples: sum={total_samples_sum} mean={total_samples_mean:.2f} std={total_samples_std:.2f}",
                                f"unique_motif_signatures={dfa['signature'].nunique()}",
                                f"variance_summary: mean_std={conc_std_mean:.6f} mean_cv={conc_cv_mean:.6f}",
                                f"plots: {plots_dir.relative_to(args.output_dir)}/top{top_n}_horizontal.png, top{top_n}_boxplot.png" + (", triads_bar.png" if (cfg.directed and k == 3) else ""),
                                f"per-signature CSV: {sig_csv.relative_to(args.output_dir)}",
                            ]
                            + (["Top motifs (by mean concentration):"] + top_lines if top_lines else [])
                            + ((["Triad aggregates:"] + tri_section_lines) if tri_section_lines else [])
                        )
                    )
                except Exception as e:
                    summary_lines.append(f"[WARN] Aggregated summary failed for k={k}: {e}")

        # Write dataset summary text file
        ds_summary = args.output_dir / ds_key / "summary.txt"
        t_sum0 = time.time()
        with open(ds_summary, "w") as f:
            f.write("\n".join(summary_lines) + "\n")
        t_sum1 = time.time()
        print(f"[phase.summary] wrote {ds_summary.relative_to(args.output_dir)} in {t_sum1 - t_sum0:.2f}s", flush=True)
        
        # Dataset completion summary
        ds_elapsed = time.time() - t0
        print(f"\n{'─'*70}")
        print(f"✓ COMPLETED DATASET: {ds_key}")
        print(f"  Total time: {ds_elapsed:.1f}s ({ds_elapsed/60:.1f} min)")
        print(f"  Graph: n={G.number_of_nodes()} m={G.number_of_edges()}")
        print(f"  Runs completed: {len(args.k) * len(args.q) * len(args.seed)}")
        print(f"{'─'*70}\n")
    
    # Overall completion summary
    t_all_end = time.time()
    total_elapsed = t_all_end - t_all0
    print("\n" + "="*70)
    print("✓ ALL EXPERIMENTS COMPLETED")
    print("="*70)
    print(f"Total datasets:  {len(args.datasets)}")
    print(f"Total runs:      {total_runs}")
    print(f"Total time:      {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Avg per run:     {total_elapsed/total_runs:.2f}s" if total_runs > 0 else "")
    print(f"Output dir:      {args.output_dir}")
    print("="*70 + "\n")
    
    # CONTRIBUTION: Performance scaling analysis across datasets
    print("\n" + "="*70)
    print("CONTRIBUTION: Performance Scaling Analysis")
    print("="*70)
    print("Analyzing how runtime scales with network size (n, m)...")
    
    scaling_data = []
    for ds_key in args.datasets:
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
                
                scaling_data.append({
                    "dataset": ds_key,
                    "k": k,
                    "n": n,
                    "m": m,
                    "avg_runtime_sec": avg_runtime,
                    "avg_samples": avg_samples,
                    "samples_per_sec": samples_per_sec,
                    "runtime_per_edge": avg_runtime / m if m > 0 else 0,
                })
    
    if scaling_data:
        scaling_df = pd.DataFrame(scaling_data)
        scaling_csv = args.output_dir / "performance_scaling.csv"
        scaling_df.to_csv(scaling_csv, index=False)
        print(f"✓ Saved scaling analysis: {scaling_csv}")
        
        print("\nPerformance Scaling Summary:")
        print("-" * 70)
        for k in sorted(scaling_df["k"].unique()):
            k_data = scaling_df[scaling_df["k"] == k].sort_values("n")
            print(f"\nk={k}:")
            print(f"{'Dataset':<20} {'Nodes':>10} {'Edges':>10} {'Runtime(s)':>12} {'Samples/s':>12}")
            print("-" * 70)
            for _, row in k_data.iterrows():
                print(f"{row['dataset']:<20} {row['n']:>10} {row['m']:>10} {row['avg_runtime_sec']:>12.2f} {row['samples_per_sec']:>12.1f}")
        
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
    else:
        print("[WARN] No scaling data available")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
