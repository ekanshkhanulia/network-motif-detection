from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import statistics
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd

from src.config import DATASETS, resolve_data_path
from src.utils.io import load_snap_graph


def estimate_power_law_exponent(degrees, k_min: int = 1):
    """Simple MLE for discrete power-law alpha ~ 1 + n / sum(log(k/k_min - 0.5))"""
    filtered = [k for k in degrees if k >= k_min]
    n = len(filtered)
    if n < 10:
        return None
    denom = sum(math.log(k / (k_min - 0.5)) for k in filtered)
    if denom <= 0:
        return None
    return 1 + n / denom


def compute_centrality_stats(G: nx.Graph, directed: bool, sample_size: int = 1000) -> Dict:
    """Compute centrality statistics on a sample of nodes (for large graphs).
    
    Returns mean and top-k values for degree, betweenness, and closeness centrality.
    Aligned with Lecture 2 concepts.
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    
    # For very large graphs, sample nodes for expensive metrics
    if n > sample_size:
        import random
        sampled_nodes = random.sample(nodes, min(sample_size, n))
    else:
        sampled_nodes = nodes
    
    # Degree centrality (normalized)
    if directed:
        deg_vals = [G.degree(v) / (n - 1) for v in sampled_nodes]
    else:
        deg_vals = [G.degree(v) / (n - 1) for v in sampled_nodes]
    
    # Betweenness centrality (sampled for large graphs)
    try:
        if n > 500:
            # Use k-sample approximation
            betweenness = nx.betweenness_centrality(G, k=min(100, n))
        else:
            betweenness = nx.betweenness_centrality(G)
        bet_vals = [betweenness.get(v, 0.0) for v in sampled_nodes]
    except:
        bet_vals = []
    
    # Closeness centrality (on sample only for large graphs)
    close_vals = []
    if n <= 500:
        try:
            closeness = nx.closeness_centrality(G)
            close_vals = [closeness.get(v, 0.0) for v in sampled_nodes]
        except:
            pass
    
    return {
        "degree_centrality_mean": statistics.mean(deg_vals) if deg_vals else None,
        "degree_centrality_max": max(deg_vals) if deg_vals else None,
        "betweenness_centrality_mean": statistics.mean(bet_vals) if bet_vals else None,
        "betweenness_centrality_max": max(bet_vals) if bet_vals else None,
        "closeness_centrality_mean": statistics.mean(close_vals) if close_vals else None,
        "closeness_centrality_max": max(close_vals) if close_vals else None,
    }


def compute_distance_stats(G: nx.Graph, directed: bool, sample_size: int = 500) -> Dict:
    """Compute diameter and average path length (sampled for large graphs).
    
    Aligned with Lecture 1 small-world phenomenon concepts.
    """
    n = G.number_of_nodes()
    
    # Work on largest connected component
    if directed:
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))
    
    if not components:
        return {"diameter": None, "avg_shortest_path": None}
    
    largest_cc = max(components, key=len)
    G_cc = G.subgraph(largest_cc).copy()
    
    # For large graphs, estimate using sample
    if len(largest_cc) > sample_size:
        import random
        sample_nodes = random.sample(list(largest_cc), min(sample_size, len(largest_cc)))
        G_sample = G_cc.subgraph(sample_nodes)
        
        # Convert to undirected for distance calculations
        if directed:
            G_sample = G_sample.to_undirected()
        
        if nx.is_connected(G_sample):
            try:
                diameter = nx.diameter(G_sample)
                avg_path = nx.average_shortest_path_length(G_sample)
                return {
                    "diameter_estimate": diameter,
                    "avg_shortest_path_estimate": avg_path,
                    "sampled": True,
                }
            except:
                pass
        return {"diameter_estimate": None, "avg_shortest_path_estimate": None, "sampled": True}
    else:
        # Small enough to compute exactly
        if directed:
            G_cc = G_cc.to_undirected()
        
        if nx.is_connected(G_cc):
            try:
                diameter = nx.diameter(G_cc)
                avg_path = nx.average_shortest_path_length(G_cc)
                return {"diameter": diameter, "avg_shortest_path": avg_path, "sampled": False}
            except:
                pass
    
    return {"diameter": None, "avg_shortest_path": None, "sampled": False}


def main():
    parser = argparse.ArgumentParser(description="Compute comprehensive network metrics (aligned with SNACS Lecture 1-2)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/network_metrics.json"))
    parser.add_argument("--centrality-sample", type=int, default=1000, help="Sample size for centrality on large graphs")
    parser.add_argument("--skip-centrality", action="store_true", help="Skip expensive centrality calculations")
    parser.add_argument("--skip-distance", action="store_true", help="Skip diameter/path length calculations")

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_stats: Dict[str, Dict] = {}

    for ds in args.datasets:
        cfg = DATASETS.get(ds)
        if not cfg:
            print(f"[WARN] Unknown dataset {ds}, skipping")
            continue
        
        print(f"\n[{ds}] Loading graph...")
        path = resolve_data_path(args.data_dir, ds)
        G = load_snap_graph(path, directed=cfg.directed, max_nodes=args.max_nodes)
        n = G.number_of_nodes()
        m = G.number_of_edges()
        
        print(f"[{ds}] n={n} m={m} directed={cfg.directed}")
        
        # Basic degree statistics
        if cfg.directed:
            indeg = [d for _, d in G.in_degree()]
            outdeg = [d for _, d in G.out_degree()]
            degrees = [d_in + d_out for d_in, d_out in zip(indeg, outdeg)]
            mean_indeg = statistics.mean(indeg) if indeg else 0
            mean_outdeg = statistics.mean(outdeg) if outdeg else 0
        else:
            degrees = [d for _, d in G.degree()]
            mean_indeg = None
            mean_outdeg = None
        
        mean_deg = statistics.mean(degrees) if degrees else 0
        median_deg = statistics.median(degrees) if degrees else 0
        var_deg = statistics.pvariance(degrees) if len(degrees) > 1 else 0
        max_deg = max(degrees) if degrees else 0
        
        # Power-law exponent estimation (Lecture 1 concept)
        alpha_est = estimate_power_law_exponent(degrees, k_min=1)
        
        print(f"[{ds}] mean_degree={mean_deg:.2f} max_degree={max_deg} alpha_est={alpha_est}")

        # Component analysis
        if cfg.directed:
            wcc_sizes = [len(c) for c in nx.weakly_connected_components(G)]
            scc_sizes = [len(c) for c in nx.strongly_connected_components(G)]
            largest_wcc = max(wcc_sizes) if wcc_sizes else 0
            largest_scc = max(scc_sizes) if scc_sizes else 0
            num_wcc = len(wcc_sizes)
            num_scc = len(scc_sizes)
        else:
            cc_sizes = [len(c) for c in nx.connected_components(G)]
            largest_wcc = max(cc_sizes) if cc_sizes else 0
            largest_scc = largest_wcc
            num_wcc = len(cc_sizes)
            num_scc = num_wcc

        # Clustering & triangles (Lecture 1 concept)
        if cfg.directed:
            clustering = None
            triangle_count = None
        else:
            print(f"[{ds}] Computing clustering coefficient...")
            clustering = nx.average_clustering(G)
            triangle_count = sum(nx.triangles(G).values()) // 3
            print(f"[{ds}] clustering={clustering:.4f} triangles={triangle_count}")

        # Density (Lecture 1 concept)
        if cfg.directed:
            max_edges = n * (n - 1)
        else:
            max_edges = n * (n - 1) // 2
        density = m / max_edges if max_edges > 0 else 0

        stats = {
            "directed": cfg.directed,
            "n": n,
            "m": m,
            "density": density,
            "mean_degree": mean_deg,
            "median_degree": median_deg,
            "max_degree": max_deg,
            "var_degree": var_deg,
            "mean_indegree": mean_indeg,
            "mean_outdegree": mean_outdeg,
            "power_law_alpha_est": alpha_est,
            "num_wcc": num_wcc,
            "num_scc": num_scc,
            "largest_wcc": largest_wcc,
            "largest_scc": largest_scc,
            "giant_component_fraction": largest_wcc / n if n > 0 else 0,
            "average_clustering": clustering,
            "triangle_count": triangle_count,
        }
        
        # Centrality analysis (Lecture 2 concepts)
        if not args.skip_centrality:
            print(f"[{ds}] Computing centrality metrics...")
            centrality_stats = compute_centrality_stats(G, cfg.directed, args.centrality_sample)
            stats.update(centrality_stats)
            print(f"[{ds}] degree_centrality_mean={centrality_stats.get('degree_centrality_mean', 'N/A')}")
        
        # Distance metrics (Lecture 1 small-world concept)
        if not args.skip_distance:
            print(f"[{ds}] Computing distance metrics...")
            distance_stats = compute_distance_stats(G, cfg.directed, sample_size=500)
            stats.update(distance_stats)
            print(f"[{ds}] diameter={distance_stats.get('diameter') or distance_stats.get('diameter_estimate', 'N/A')}")
        
        all_stats[ds] = stats
        print(f"[{ds}] ✓ Complete\n")

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n✓ Saved comprehensive network metrics to {args.output}")
    
    # Also create a CSV for easier reading
    csv_path = args.output.with_suffix(".csv")
    df = pd.DataFrame.from_dict(all_stats, orient="index")
    df.index.name = "dataset"
    df.to_csv(csv_path)
    print(f"✓ Saved CSV version to {csv_path}")


if __name__ == "__main__":
    main()
