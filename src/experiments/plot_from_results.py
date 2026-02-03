from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Dict, List

import pandas as pd

from src.utils.motifs import triad_label_from_signature
from src.utils.visualize import (
    plot_motif_distribution,
    plot_motif_distribution_horizontal,
    plot_seed_boxplot,
)


def load_runs(folder: Path) -> pd.DataFrame:
    records = []
    for csv in folder.glob("*.csv"):
        try:
            df = pd.read_csv(csv, dtype={"signature": str})
        except Exception:
            continue
        if df.empty:
            continue
        # Add run identifiers
        meta_path = csv.with_name(csv.stem + "_meta.json")
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
        df["run"] = csv.stem
        for k, v in meta.items():
            # Only attach scalar metadata; skip lists/dicts to avoid length mismatch
            if isinstance(v, (int, float, str, bool)):
                df[f"meta_{k}"] = v
        records.append(df)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Create nicer plots from results CSVs")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--group-triads", action="store_true", help="Group k=3 directed by triad labels")

    args = parser.parse_args()

    folder = args.results_dir / args.dataset / f"k{args.k}"
    out_dir = folder / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_runs(folder)
    if df.empty:
        print("No runs found to plot.")
        return

    # Top-N by mean concentration across runs
    grouped = (
        df.groupby("signature")["concentration"].mean().sort_values(ascending=False)
    )
    top_signatures = list(grouped.head(args.top_n).index)

    # 1) Horizontal bar of top-N mean concentrations
    top_map: Dict[str, int] = {}
    for sig in top_signatures:
        sub = df[df["signature"] == sig]
        if "total_samples" in sub.columns:
            # Reconstruct mean count via concentration * mean total
            mean_total = sub["total_samples"].mean()
            mean_cnt = float((sub["concentration"] * sub["total_samples"]).mean())
        else:
            mean_cnt = float(sub["count"].mean())
        top_map[str(sig)] = int(round(mean_cnt))

    plot_motif_distribution_horizontal(
        top_map,
        total=int(df.get("total_samples", pd.Series([1])).mean()),
        title=f"{args.dataset} k={args.k}: Top-{args.top_n} motifs (mean)",
        out_path=out_dir / f"top{args.top_n}_horizontal.png",
    )

    # 2) Boxplot of concentrations across seeds/runs for the Top-N
    rows: List[Dict[str, float]] = []
    for sig in top_signatures:
        sub = df[df["signature"] == sig]
        for _, r in sub.iterrows():
            rows.append({
                "signature": str(sig),
                "seed": r.get("seed", r.get("meta_seed", 0)),
                "concentration": r["concentration"],
            })
    plot_seed_boxplot(
        rows,
        out_path=out_dir / f"top{args.top_n}_boxplot.png",
        title=f"{args.dataset} k={args.k}: Concentration variance (Top-{args.top_n})",
    )

    # 3) Optional triad grouping for k=3 directed
    if args.group_triads and args.k == 3:
        if bool(df.get("directed", pd.Series([True])).iloc[0]):
            df_tri = df.copy()
            df_tri["triad"] = df_tri["signature"].apply(triad_label_from_signature)
            triad_mean = df_tri.groupby("triad")["concentration"].mean().sort_values(ascending=False)
            triad_map = {triad: int(round(c * 1000)) for triad, c in triad_mean.items()}  # pseudo-counts for plotting
            plot_motif_distribution(
                triad_map,
                total=sum(triad_map.values()),
                title=f"{args.dataset} k=3: Triad distribution (mean)",
                out_path=out_dir / "triads_bar.png",
                top_n=len(triad_map),
            )

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
