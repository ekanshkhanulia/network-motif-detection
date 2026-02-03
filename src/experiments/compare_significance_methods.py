from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.config import DATASETS
from src.experiments.significance_edge_swaps import run_edge_swap_significance
from src.experiments.significance_direct_bender_canfield import run_direct_significance_bc
from src.utils.visualize import plot_scatter_xy, plot_significance_scatter


def main():
    parser = argparse.ArgumentParser(description="Compare ensemble vs direct significance for directed triads (k=3)")
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--q", type=float, default=0.01, help="Sampling fraction used in RAND-ESU comparisons")
    parser.add_argument("--schedule", type=str, default="skewed", help="RAND-ESU schedule name")
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3], help="Seeds for original runs")
    parser.add_argument("--n-rand", type=int, default=200, help="Number of randomized graphs (edge swaps method)")
    parser.add_argument("--swap-iters", type=int, default=2, help="Edge swaps multiplier over m (per random graph)")
    parser.add_argument("--direct-T", type=int, default=20000, help="Monte Carlo triples sampled for direct method")
    parser.add_argument("--direct-seed", type=int, default=1, help="Random seed for direct method Monte Carlo")
    parser.add_argument("--no-scatter", action="store_true", help="Skip scatter plot generation")

    args = parser.parse_args()
    cfg = DATASETS[args.dataset]
    if not cfg.directed:
        raise SystemExit("This comparison targets directed graphs (triads) only.")

    out_dir = args.output_dir / args.dataset / "k3" / "significance_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run ensemble method and capture CSV path by calling the module's main via CLI args
    t0 = time.time()
    edge_swap_result = run_edge_swap_significance(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        k=3,
        q=args.q,
        schedule=args.schedule,
        seeds=args.seeds,
        random_graphs=args.n_rand,
        swaps_per_edge=args.swap_iters,
        max_nodes=args.max_nodes,
        use_existing_rand_esu=True,  # Load from existing CSVs for consistency
    )
    t1 = time.time()
    ens_csv = edge_swap_result["csv_path"]

    # For article-aligned baseline only: copy or summarize ensemble CSV as the reference baseline
    ens = edge_swap_result["df"].copy()
    # Ensure triad column for readability
    if "triad" not in ens.columns and "signature" in ens.columns:
        from src.utils.motifs import triad_label_from_signature
        ens["triad"] = ens["signature"].astype(str).apply(triad_label_from_signature)
    # Save a baseline-only summary CSV
    out_csv = out_dir / "significance_ensemble_baseline.csv"
    ens.to_csv(out_csv, index=False)
    # And a concise txt summary (no markdown)
    out_txt = out_dir / "significance_ensemble_summary.txt"
    with open(out_txt, "w") as f:
        f.write(f"Ensemble baseline (edge swaps) for {args.dataset}, triads (k=3)\n")
        f.write(
            f"Graphs randomized: {args.n_rand}, swaps_per_edge multiplier: {args.swap_iters}, seeds={args.seeds}\n"
        )
        f.write(f"q={args.q} schedule={args.schedule} runtime={t1 - t0:.2f}s\n")
        f.write(
            "Metrics: enrichment_ratio=orig_concentration/rand_mean_concentration (paper-aligned). "
            "Z-scores are included for reference and can be extreme when rand_std_concentration is tiny.\n"
        )
        if "triad" in ens.columns and "rand_mean_concentration" in ens.columns:
            top = ens.sort_values("rand_mean_concentration", ascending=False).head(10)
            f.write("Top-10 triads by mean concentration in ensemble:\n")
            for _, r in top.iterrows():
                f.write(
                    f"  {r['triad']}: mean={r['rand_mean_concentration']:.6g} std={r.get('rand_std_concentration', float('nan')):.6g}\n"
                )
            if "enrichment_ratio" in ens.columns:
                top_ratio = ens.sort_values("enrichment_ratio", ascending=False).head(10)
                f.write("\nTop-10 triads by enrichment ratio (orig vs ensemble mean):\n")
                for _, r in top_ratio.iterrows():
                    f.write(
                        f"  {r['triad']}: ratio={r['enrichment_ratio']:.6g} orig={r['orig_concentration']:.6g} rand_mean={r['rand_mean_concentration']:.6g}\n"
                    )
    print(f"Saved ensemble baseline CSV and summary txt to {out_dir}")

    # Run direct method prototype (Chung-Lu expectation approximation)
    print("Running direct significance (Bender–Canfield inspired) ...")
    direct_result = run_direct_significance_bc(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        T=args.direct_T,
        seed=args.direct_seed,
        max_nodes=args.max_nodes,
    )
    direct_df = direct_result["df"].copy()
    
    # Check if direct method produced any results
    if direct_df.empty or "signature" not in direct_df.columns:
        print(f"[WARN] Direct BC method produced no rows. Skipping comparison merge.")
        warn_txt = out_dir / "significance_direct_bc_unavailable.txt"
        with open(warn_txt, "w") as f:
            f.write("Direct BC significance produced no rows (possibly T too small or no connected triads found).\n")
            f.write("Skipping ensemble vs direct comparison merge and plots.\n")
        return

    # Rename columns for join clarity
    direct_df.rename(
        columns={
            "observed_concentration": "direct_bc_observed_concentration",
            "expected_concentration": "direct_bc_expected_concentration",
            "enrichment_ratio": "direct_bc_enrichment_ratio",
            "z_score": "direct_bc_z_score",
        },
        inplace=True,
    )

    # Merge ensemble (edge-swap) and direct expectation on signature
    merged = pd.merge(
        ens,
        direct_df,
        how="inner",
        left_on="signature",
        right_on="signature",
    )

    merged_out = out_dir / "significance_ensemble_vs_direct.csv"
    merged.to_csv(merged_out, index=False)
    print(f"Saved merged ensemble vs direct comparison: {merged_out}")

    # Scatter plots (ensemble mean vs direct expected; original vs direct expected)
    if not args.no_scatter:
        try:
            scatter1 = out_dir / "scatter_rand_mean_vs_direct_expected_bc.png"
            plot_scatter_xy(
                merged,
                x_col="rand_mean_concentration",
                y_col="direct_bc_expected_concentration",
                color_col="direct_bc_z_score",
                out_path=scatter1,
                title=f"{args.dataset} Triads: Ensemble mean vs Direct-BC expected",
                x_label="Ensemble mean concentration",
                y_label="Direct-BC expected concentration",
            )
            scatter2 = out_dir / "scatter_orig_vs_direct_expected_bc.png"
            plot_scatter_xy(
                merged,
                x_col="orig_concentration",
                y_col="direct_bc_expected_concentration",
                color_col="direct_bc_z_score",
                out_path=scatter2,
                title=f"{args.dataset} Triads: Original vs Direct-BC expected",
                x_label="Original concentration (RAND-ESU)",
                y_label="Direct-BC expected concentration",
            )
            # Colored z-score scatter using existing helper (orig vs ensemble mean)
            zscatter = out_dir / "scatter_orig_vs_ensemble_mean.png"
            plot_significance_scatter(
                ens.rename(columns={"orig_concentration": "orig_concentration"}),
                out_path=zscatter,
                title=f"{args.dataset} Triads: Original vs Ensemble (z-colored)",
            )
            print("Saved scatter plots (ensemble vs direct; original vs direct; z-colored original vs ensemble).")
        except Exception as exc:
            print(f"[WARN] Failed to generate scatter plots: {exc}")

    # Append comparison summary to txt
    comp_txt = out_dir / "significance_comparison_summary.txt"
    with open(comp_txt, "w") as f:
        f.write("Significance comparison (edge-swaps ensemble vs direct BC-inspired)\n")
        f.write(f"Dataset={args.dataset} k=3 T_direct={args.direct_T} direct_seed={args.direct_seed}\n")
        f.write(f"Ensemble: random_graphs={args.n_rand} swaps_per_edge={args.swap_iters} q={args.q} schedule={args.schedule}\n")
        f.write(
            "Metrics: enrichment_ratio columns are paper-aligned; Z-scores are provided for reference and can be extreme when variance is tiny.\n\n"
        )
        # Simple stats: correlation between ensemble mean and direct expected
        try:
            corr = merged[["rand_mean_concentration", "direct_bc_expected_concentration"]].corr().iloc[0,1]
            f.write(f"Pearson corr (ensemble mean vs direct-BC expected): {corr:.4f}\n")
        except Exception:
            f.write("Pearson corr (ensemble mean vs direct-BC expected): NA\n")
        # Top 5 by absolute difference
        merged["abs_diff"] = (merged["rand_mean_concentration"] - merged["direct_bc_expected_concentration"]).abs()
        top_diff = merged.sort_values("abs_diff", ascending=False).head(5)
        f.write("\nTop 5 triads by |ensemble_mean - direct_expected|:\n")
        for _, r in top_diff.iterrows():
            triad = r.get("triad", r["signature"])
            f.write(
                f"  {triad}: ensemble_mean={r['rand_mean_concentration']:.6g} direct_expected={r['direct_bc_expected_concentration']:.6g} diff={r['abs_diff']:.6g}\n"
            )
    print(f"Saved comparison summary: {comp_txt}")


if __name__ == "__main__":
    main()
