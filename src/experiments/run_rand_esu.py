from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import List

import networkx as nx
import pandas as pd

from src.config import DATASETS, resolve_data_path
from src.utils.io import load_snap_graph
from src.utils.motifs import count_motif_signatures
from src.algorithms.rand_esu import RandESUParams, rand_esu_sample
from src.experiments.common import build_p_schedule
from src.utils.visualize import plot_motif_distribution



def main():
    parser = argparse.ArgumentParser(description="Run RAND-ESU motif sampling on SNAP datasets")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()), help="Dataset keys to run")
    parser.add_argument("--k", type=int, nargs="*", default=[3, 4, 5], help="Motif sizes")
    parser.add_argument("--q", type=float, default=0.01, help="Expected fraction of subgraphs sampled (prod p_d)")
    parser.add_argument("--schedule", type=str, default="skewed", choices=["geometric", "skewed"], help="Depth prob schedule")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-nodes", type=int, default=None, help="Optional node cap for smoke tests")
    parser.add_argument("--child-selection", type=str, default="bernoulli", choices=["bernoulli", "balanced"], help="Child selection strategy")

    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
        print(f"[phase.load] n={G.number_of_nodes()} m={G.number_of_edges()} took {t1 - t0:.2f}s", flush=True)

        # For connectivity in ESU on directed graphs, use weak connectivity; ESU uses neighbors per directed edges
        # but connectivity constraint is enforced by ESU construction; no extra handling needed here.

        for k in args.k:
            p_depth = build_p_schedule(k, args.q, args.schedule)
            print(f"[run.start] k={k} q={args.q} schedule={args.schedule} child={args.child_selection} -> p={p_depth}", flush=True)

            # Set seed
            import random
            random.seed(args.seed)

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
                        "q": args.q,
                        "schedule": args.schedule,
                        "seed": args.seed,
                        "total_samples": total,
                        "signature": sig,
                        "count": count,
                        "concentration": (count / total) if total > 0 else 0.0,
                    }
                )
            df = pd.DataFrame.from_records(records)
            # Organized subdirectories: results/dataset/k{k}/
            subdir = args.output_dir / ds_key / f"k{k}"
            subdir.mkdir(parents=True, exist_ok=True)
            csv_path = subdir / f"q{args.q}_{args.schedule}_seed{args.seed}.csv"
            t_save0 = time.time()
            df.to_csv(csv_path, index=False)

            # Prefer relative file path (to data dir) in meta
            try:
                rel_file = str(path.resolve().relative_to(args.data_dir.resolve()))
            except Exception:
                rel_file = Path(path).name

            meta = {
                "dataset": ds_key,
                "file": rel_file,
                "directed": cfg.directed,
                "n": G.number_of_nodes(),
                "m": G.number_of_edges(),
                "k": k,
                "q": args.q,
                "p_depth": p_depth,
                "schedule": args.schedule,
                "seed": args.seed,
                "runtime_sec": t3 - t2,
                "loaded_sec": t1 - t0,
                "total_samples": total,
                "unique_motif_classes": len(freq),
            }
            json_path = subdir / f"q{args.q}_{args.schedule}_seed{args.seed}_meta.json"
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=2)
            # Plot motif distribution
            plot_path = subdir / f"q{args.q}_{args.schedule}_seed{args.seed}_motifs.png"
            plot_title = f"{ds_key} k={k} q={args.q} seed={args.seed} sel={args.child_selection}"
            plot_motif_distribution(freq, total, plot_title, plot_path)
            t_save1 = time.time()
            print(f"  [phase.save] csv+json+plot took {t_save1 - t_save0:.2f}s -> {csv_path.name}", flush=True)


if __name__ == "__main__":
    main()
