"""
Compare Bender-Canfield direct method ⟨Ĉ⟩ vs Edge-Swap ensemble method ⟨C⟩.

This implements the comparison shown in Wernicke (2005) Table 2:
- ⟨C⟩ = mean concentration from edge-swap ensemble (10,000 random graphs)
- ⟨Ĉ⟩ = expected concentration from BC direct method (100,000 samples)
- ratio = ⟨C⟩ / ⟨Ĉ⟩ (should be close to 1 for accurate approximation)

The key insight is that both methods estimate the EXPECTED concentration in random
graphs with the same degree sequence. The BC method is ~100x faster.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config import DATASETS
from src.utils.motifs import triad_label_from_signature


def load_bc_results(dataset: str, results_dir: Path) -> Dict[str, float]:
    """Load BC expected concentrations from results."""
    bc_dir = results_dir / dataset / "k3" / "significance_direct_bc"

    # Find most recent BC results
    csv_files = list(bc_dir.glob("direct_bc_exact_*.csv"))
    if not csv_files:
        return {}

    # Use the one with highest T
    csv_files.sort(key=lambda x: int(x.stem.split("_T")[1].split("_")[0]), reverse=True)
    bc_csv = csv_files[0]

    df = pd.read_csv(bc_csv, dtype={"signature": str})
    return {row["signature"]: row["expected_concentration"] for _, row in df.iterrows()}


def load_edgeswap_results(dataset: str, results_dir: Path) -> Dict[str, float]:
    """Load edge-swap ensemble concentrations from results."""
    sig_dir = results_dir / dataset / "k3" / "significance"

    csv_files = list(sig_dir.glob("significance_*.csv"))
    if not csv_files:
        return {}

    df = pd.read_csv(csv_files[0], dtype={"signature": str})
    return {
        row["signature"]: row["rand_mean_concentration"] for _, row in df.iterrows()
    }


def load_observed_concentrations(dataset: str, results_dir: Path) -> Dict[str, float]:
    """Load observed concentrations from original graph."""
    sig_dir = results_dir / dataset / "k3" / "significance"

    csv_files = list(sig_dir.glob("significance_*.csv"))
    if not csv_files:
        return {}

    df = pd.read_csv(csv_files[0], dtype={"signature": str})
    return {row["signature"]: row["orig_concentration"] for _, row in df.iterrows()}


def compare_methods(
    dataset: str,
    bc_dir: Path,
    edgeswap_dir: Path,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Compare BC direct method vs edge-swap ensemble method.

    Returns DataFrame with columns:
    - signature, triad: pattern identifiers
    - observed: C_k^i(G) from original graph
    - edgeswap: ⟨C⟩ from edge-swap ensemble
    - bc: ⟨Ĉ⟩ from BC direct method
    - ratio: ⟨C⟩ / ⟨Ĉ⟩
    """
    cfg = DATASETS.get(dataset)
    if cfg is None:
        raise ValueError(f"Unknown dataset: {dataset}")

    bc_exp = load_bc_results(dataset, bc_dir)
    es_exp = load_edgeswap_results(dataset, edgeswap_dir)
    obs = load_observed_concentrations(dataset, edgeswap_dir)

    if not bc_exp:
        print(f"[WARN] No BC results found for {dataset}")
        return pd.DataFrame()
    if not es_exp:
        print(f"[WARN] No edge-swap results found for {dataset}")
        return pd.DataFrame()

    # Combine all signatures
    all_sigs = set(bc_exp.keys()) | set(es_exp.keys()) | set(obs.keys())

    rows = []
    for sig in sorted(all_sigs):
        bc_c = bc_exp.get(sig, 0.0)
        es_c = es_exp.get(sig, 0.0)
        obs_c = obs.get(sig, 0.0)

        # Compute ratio ⟨C⟩ / ⟨Ĉ⟩
        ratio = es_c / bc_c if bc_c > 1e-10 else float("nan")

        rows.append(
            {
                "dataset": dataset,
                "signature": sig,
                "triad": triad_label_from_signature(sig)
                if cfg.directed
                else "undirected",
                "observed": obs_c,
                "edgeswap_mean": es_c,
                "bc_expected": bc_c,
                "ratio_C_over_Chat": ratio,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("observed", ascending=False)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"Comparison: Edge-Swap ⟨C⟩ vs BC Direct ⟨Ĉ⟩ for {dataset}")
    print(f"{'=' * 70}")
    print(f"{'Triad':<8} {'Observed':>10} {'⟨C⟩':>10} {'⟨Ĉ⟩':>10} {'Ratio':>8}")
    print("-" * 70)

    for _, row in df.iterrows():
        if row["observed"] > 1e-5:  # Only show significant patterns
            print(
                f"{row['triad']:<8} {row['observed']:>10.4f} "
                f"{row['edgeswap_mean']:>10.4f} {row['bc_expected']:>10.4f} "
                f"{row['ratio_C_over_Chat']:>8.2f}"
            )

    # Summary statistics
    ratios = df[df["observed"] > 1e-5]["ratio_C_over_Chat"].dropna()
    if len(ratios) > 0:
        print("-" * 70)
        print(f"Ratio statistics (for patterns with obs > 1e-5):")
        print(f"  Mean ratio: {ratios.mean():.3f}")
        print(f"  Std ratio: {ratios.std():.3f}")
        print(f"  Min ratio: {ratios.min():.3f}")
        print(f"  Max ratio: {ratios.max():.3f}")

    # Save if output directory provided
    if output_dir is not None:
        out_dir = output_dir / dataset / "k3" / "method_comparison"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "bc_vs_edgeswap.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nSaved to: {out_csv}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compare BC direct method vs edge-swap ensemble (Wernicke 2005 Table 2)"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=list(DATASETS.keys())
    )
    parser.add_argument(
        "--bc-dir",
        type=Path,
        default=Path("results_bc_significance"),
        help="Directory with BC results",
    )
    parser.add_argument(
        "--edgeswap-dir",
        type=Path,
        default=Path("results_baseline_esa_50k_samples"),
        help="Directory with edge-swap results",
    )
    parser.add_argument("--output-dir", type=Path, default=None)

    args = parser.parse_args()
    compare_methods(
        dataset=args.dataset,
        bc_dir=args.bc_dir,
        edgeswap_dir=args.edgeswap_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
