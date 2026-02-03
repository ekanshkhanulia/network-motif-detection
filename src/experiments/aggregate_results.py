from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd
import statistics
from typing import Dict, List


def compute_motif_diversity(df: pd.DataFrame) -> float:
    """Compute motif diversity (Shannon entropy) for a given result set."""
    if df.empty or "concentration" not in df.columns:
        return 0.0
    
    concentrations = df["concentration"].values
    # Filter out zeros
    concentrations = [c for c in concentrations if c > 0]
    if not concentrations:
        return 0.0
    
    import math
    entropy = -sum(c * math.log(c) for c in concentrations if c > 0)
    return entropy


def analyze_cross_dataset_patterns(results_dir: Path, k: int) -> pd.DataFrame:
    """Analyze which motifs are universal vs dataset-specific.
    
    Returns DataFrame with motif signatures and their prevalence across datasets.
    """
    all_motifs = {}
    dataset_results = {}
    
    for dataset_dir in results_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        k_dir = dataset_dir / f"k{k}"
        if not k_dir.exists():
            continue
        
        dataset_name = dataset_dir.name
        motif_data = []
        
        for csv in k_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv)
                if not df.empty and "signature" in df.columns and "concentration" in df.columns:
                    motif_data.append(df[["signature", "concentration"]])
            except:
                continue
        
        if motif_data:
            combined = pd.concat(motif_data, ignore_index=True)
            mean_conc = combined.groupby("signature")["concentration"].mean()
            dataset_results[dataset_name] = mean_conc.to_dict()
            
            for sig in mean_conc.index:
                if sig not in all_motifs:
                    all_motifs[sig] = []
                all_motifs[sig].append(dataset_name)
    
    # Create summary
    rows = []
    for sig, datasets in all_motifs.items():
        num_datasets = len(datasets)
        mean_concs = [dataset_results[ds].get(sig, 0.0) for ds in datasets]
        rows.append({
            "signature": sig,
            "num_datasets": num_datasets,
            "datasets": ",".join(sorted(datasets)),
            "mean_concentration": statistics.mean(mean_concs),
            "std_concentration": statistics.stdev(mean_concs) if len(mean_concs) > 1 else 0.0,
            "universality": num_datasets / len(dataset_results) if dataset_results else 0.0,
        })
    
    return pd.DataFrame(rows).sort_values("universality", ascending=False)


def correlate_motifs_with_properties(results_dir: Path, network_metrics_path: Path, k: int) -> pd.DataFrame:
    """Correlate motif patterns with network properties.
    
    Analyzes relationship between network metrics (power-law exponent, clustering, etc.)
    and motif diversity/distribution.
    """
    # Load network metrics
    if not network_metrics_path.exists():
        print(f"[WARN] Network metrics file not found: {network_metrics_path}")
        return pd.DataFrame()
    
    with open(network_metrics_path, "r") as f:
        metrics = json.load(f)
    
    rows = []
    for dataset_name, net_props in metrics.items():
        k_dir = results_dir / dataset_name / f"k{k}"
        if not k_dir.exists():
            continue
        
        # Aggregate motif data
        motif_data = []
        for csv in k_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv)
                if not df.empty:
                    motif_data.append(df)
            except:
                continue
        
        if not motif_data:
            continue
        
        combined = pd.concat(motif_data, ignore_index=True)
        
        # Compute motif statistics
        motif_diversity = compute_motif_diversity(combined)
        unique_motifs = combined["signature"].nunique()
        
        # For directed k=3, count specific triad types
        if k == 3 and net_props.get("directed", False):
            from src.utils.motifs import triad_label_from_signature
            combined["triad"] = combined["signature"].astype(str).apply(triad_label_from_signature)
            triad_counts = combined.groupby("triad")["concentration"].mean()
            
            # Triangular motifs (high clustering)
            triangle_conc = triad_counts.get("030T", 0.0) + triad_counts.get("120D", 0.0) + triad_counts.get("120U", 0.0)
            # Linear/chain motifs
            chain_conc = triad_counts.get("012", 0.0) + triad_counts.get("102", 0.0)
        else:
            triangle_conc = None
            chain_conc = None
        
        row = {
            "dataset": dataset_name,
            "k": k,
            "motif_diversity": motif_diversity,
            "unique_motifs": unique_motifs,
            "triangle_motif_concentration": triangle_conc,
            "chain_motif_concentration": chain_conc,
            # Network properties from Lecture 1
            "power_law_alpha": net_props.get("power_law_alpha_est"),
            "clustering_coefficient": net_props.get("average_clustering"),
            "mean_degree": net_props.get("mean_degree"),
            "density": net_props.get("density"),
            "giant_component_fraction": net_props.get("giant_component_fraction"),
            "diameter": net_props.get("diameter") or net_props.get("diameter_estimate"),
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Aggregate motif CSVs and analyze patterns (CONTRIBUTION: motif-property correlations)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--dataset", type=str, default=None, help="Specific dataset or None for all")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--network-metrics", type=Path, default=Path("results/network_metrics.json"),
                       help="Network metrics JSON for correlation analysis")
    parser.add_argument("--cross-dataset-analysis", action="store_true",
                       help="Analyze motif universality across datasets")
    parser.add_argument("--correlation-analysis", action="store_true",
                       help="Correlate motifs with network properties (Lecture 1-2 concepts)")

    args = parser.parse_args()
    
    if args.output is None:
        if args.dataset:
            args.output = args.results_dir / f"{args.dataset}_k{args.k}_aggregated.csv"
        else:
            args.output = args.results_dir / f"all_datasets_k{args.k}_aggregated.csv"

    # Original aggregation functionality
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = [d.name for d in args.results_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    records = []
    for ds in datasets:
        folder = args.results_dir / ds / f"k{args.k}"
        if not folder.exists():
            print(f"[WARN] Folder does not exist: {folder}")
            continue
        
        for csv in folder.glob("*.csv"):
            try:
                df = pd.read_csv(csv)
                if df.empty:
                    continue
                # Include run parameters from meta file if present
                meta_path = csv.with_name(csv.stem + "_meta.json")
                params = {}
                if meta_path.exists():
                    try:
                        params = json.loads(meta_path.read_text())
                    except Exception:
                        params = {}
                # Keep necessary columns
                required_cols = ["signature", "count", "concentration"]
                optional_cols = ["total_samples", "dataset", "k", "q", "seed", "schedule", "directed"]
                
                cols_to_keep = [c for c in (required_cols + optional_cols) if c in df.columns]
                df_small = df[cols_to_keep].copy()
                df_small["run"] = csv.stem
                df_small["dataset"] = ds  # Ensure dataset is set
                
                for kx, vx in params.items():
                    if kx not in df_small.columns:
                        df_small[f"meta_{kx}"] = vx
                records.append(df_small)
            except Exception as e:
                print(f"[WARN] Failed to process {csv}: {e}")
                continue

    if records:
        out = pd.concat(records, ignore_index=True)
        out.to_csv(args.output, index=False)
        print(f"✓ Saved aggregate CSV: {args.output}")
    else:
        print("No CSVs found to aggregate.")
        return
    
    # CONTRIBUTION: Cross-dataset universality analysis
    if args.cross_dataset_analysis:
        print("\n[CONTRIBUTION] Analyzing motif universality across datasets...")
        universality_df = analyze_cross_dataset_patterns(args.results_dir, args.k)
        
        if not universality_df.empty:
            univ_path = args.results_dir / f"motif_universality_k{args.k}.csv"
            universality_df.to_csv(univ_path, index=False)
            print(f"✓ Saved motif universality analysis: {univ_path}")
            
            # Print summary
            print("\nMost universal motifs (appear in most datasets):")
            print(universality_df.head(10).to_string(index=False))
            
            print("\nDataset-specific motifs (appear in fewest datasets):")
            print(universality_df.tail(10).to_string(index=False))
        else:
            print("[WARN] No data for cross-dataset analysis")
    
    # CONTRIBUTION: Correlation with network properties
    if args.correlation_analysis:
        print("\n[CONTRIBUTION] Correlating motif patterns with network properties (Lecture 1-2)...")
        corr_df = correlate_motifs_with_properties(args.results_dir, args.network_metrics, args.k)
        
        if not corr_df.empty:
            corr_path = args.results_dir / f"motif_property_correlations_k{args.k}.csv"
            corr_df.to_csv(corr_path, index=False)
            print(f"✓ Saved motif-property correlations: {corr_path}")
            
            # Compute and display correlations
            print("\nCorrelation Analysis (motif patterns vs network properties):")
            print("="*70)
            
            numeric_cols = corr_df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 1:
                correlations = corr_df[numeric_cols].corr()
                
                # Key correlations of interest
                if "motif_diversity" in correlations.columns:
                    print("\nMotif Diversity correlations:")
                    div_corr = correlations["motif_diversity"].sort_values(ascending=False)
                    for col, val in div_corr.items():
                        if col != "motif_diversity" and abs(val) > 0.1:
                            print(f"  {col:40s}: {val:6.3f}")
                
                if "power_law_alpha" in correlations.columns:
                    print("\nPower-law exponent correlations:")
                    alpha_corr = correlations["power_law_alpha"].sort_values(ascending=False)
                    for col, val in alpha_corr.items():
                        if col != "power_law_alpha" and abs(val) > 0.1:
                            print(f"  {col:40s}: {val:6.3f}")
                
                if "clustering_coefficient" in correlations.columns:
                    print("\nClustering coefficient correlations:")
                    clust_corr = correlations["clustering_coefficient"].sort_values(ascending=False)
                    for col, val in clust_corr.items():
                        if col != "clustering_coefficient" and abs(val) > 0.1:
                            print(f"  {col:40s}: {val:6.3f}")
            
            print("\nDataset Summary:")
            print(corr_df.to_string(index=False))
        else:
            print("[WARN] No data for correlation analysis. Ensure network_metrics.json exists.")


if __name__ == "__main__":
    main()
