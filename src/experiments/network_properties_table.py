from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import DATASETS, resolve_data_path
from src.utils.io import load_snap_graph
from src.algorithms.rand_esu import esu_enumerate
from src.utils.motifs import count_motif_signatures


def main():
    parser = argparse.ArgumentParser(description="Produce a Table-1-like network properties table (small k)")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    parser.add_argument("--k", type=int, nargs="*", default=[3, 4])
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))

    args = parser.parse_args()
    rows = []
    for ds in args.datasets:
        cfg = DATASETS[ds]
        path = resolve_data_path(args.data_dir, ds)
        G = load_snap_graph(path, directed=cfg.directed, max_nodes=args.max_nodes)
        row: Dict[str, object] = {
            "dataset": ds,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        }
        for k in args.k:
            try:
                subs = list(esu_enumerate(G, k))
                total, freq = count_motif_signatures(G, subs)
                row[f"size{k}_subgraphs"] = total
                row[f"size{k}_classes"] = len(freq)
            except Exception:
                row[f"size{k}_subgraphs"] = None
                row[f"size{k}_classes"] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = args.output_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "table1_like_network_properties.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved table to {out_csv}")


if __name__ == "__main__":
    main()
