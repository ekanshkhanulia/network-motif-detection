from __future__ import annotations

import argparse
from pathlib import Path
import json
import time

import networkx as nx

from src.config import DATASETS, resolve_data_path
from src.utils.io import load_snap_graph


def main():
    parser = argparse.ArgumentParser(description="Compute basic graph stats for SNAP datasets")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/graph_stats.json"))

    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_stats = {}

    for ds_key in args.datasets:
        cfg = DATASETS.get(ds_key)
        if cfg is None:
            print(f"[WARN] Unknown dataset key '{ds_key}', skipping")
            continue
        path = resolve_data_path(args.data_dir, ds_key)
        t0 = time.time()
        G = load_snap_graph(path, directed=cfg.directed, max_nodes=args.max_nodes)
        t1 = time.time()
        n = G.number_of_nodes()
        m = G.number_of_edges()
        if cfg.directed:
            # Weakly connected
            wcc_sizes = [len(c) for c in nx.weakly_connected_components(G)]
            scc_sizes = [len(c) for c in nx.strongly_connected_components(G)]
            largest_wcc = max(wcc_sizes) if wcc_sizes else 0
            largest_scc = max(scc_sizes) if scc_sizes else 0
        else:
            cc_sizes = [len(c) for c in nx.connected_components(G)]
            largest_wcc = max(cc_sizes) if cc_sizes else 0
            largest_scc = largest_wcc

        stats = {
            "directed": cfg.directed,
            "n": n,
            "m": m,
            "largest_wcc": largest_wcc,
            "largest_scc": largest_scc,
            "load_time_sec": t1 - t0,
        }
        all_stats[ds_key] = stats
        print(ds_key, stats)

    with open(args.output, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Saved stats to {args.output}")


if __name__ == "__main__":
    main()
