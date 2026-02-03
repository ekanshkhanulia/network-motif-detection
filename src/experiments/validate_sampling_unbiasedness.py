from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path
from typing import List, Dict

import networkx as nx
import pandas as pd

from src.algorithms.rand_esu import RandESUParams, esu_enumerate, rand_esu_sample
from src.experiments.common import build_p_schedule


def build_synthetic_graph(n: int, p: float, directed: bool) -> nx.Graph:
    G = nx.gnp_random_graph(n, p, directed=directed)
    # remove self-loops if any
    if directed:
        G.remove_edges_from([(u, v) for u, v in list(G.edges()) if u == v])
    return G


def exact_concentrations(G: nx.Graph, k: int) -> Dict[str, float]:
    from src.utils.motifs import count_motif_signatures
    subs = list(esu_enumerate(G, k))
    total, freq = count_motif_signatures(G, subs)
    return {sig: cnt / total for sig, cnt in freq.items()}


def run_sampling_trials(G: nx.Graph, k: int, q: float, schedule: str, child_selection: str, seeds: List[int]):
    from src.utils.motifs import count_motif_signatures
    concentrations_runs: List[Dict[str, float]] = []
    all_signatures = set()
    for seed in seeds:
        random.seed(seed)
        p_depth = build_p_schedule(k, q, schedule)
        params = RandESUParams(k=k, p_depth=p_depth, child_selection=child_selection)
        samples = rand_esu_sample(G, params)
        total, freq = count_motif_signatures(G, samples)
        conc = {sig: (cnt / total) for sig, cnt in freq.items()} if total > 0 else {}
        concentrations_runs.append(conc)
        all_signatures.update(conc.keys())
    # Align runs
    rows = []
    for sig in all_signatures:
        vals = [run.get(sig, 0.0) for run in concentrations_runs]
        rows.append({
            "signature": sig,
            "mean_concentration": statistics.mean(vals),
            "stdev_concentration": statistics.pstdev(vals),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Validate unbiasedness by comparing to full enumeration")
    parser.add_argument("--n", type=int, default=60, help="Synthetic graph nodes")
    parser.add_argument("--p", type=float, default=0.05, help="Edge probability for G(n,p)")
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--q-values", type=float, nargs="*", default=[0.001, 0.005, 0.01, 0.02])
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    parser.add_argument("--schedule", type=str, default="skewed")
    parser.add_argument("--child-selection", type=str, default="bernoulli", choices=["bernoulli", "balanced"])
    parser.add_argument("--output", type=Path, default=Path("results/unbiasedness_validation.csv"))

    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    G = build_synthetic_graph(args.n, args.p, directed=args.directed)
    print(f"Synthetic graph: n={G.number_of_nodes()} m={G.number_of_edges()} directed={args.directed}")

    exact = exact_concentrations(G, args.k)
    print(f"Exact subgraph classes: {len(exact)}")

    all_rows = []
    for q in args.q_values:
        df_runs = run_sampling_trials(G, args.k, q, args.schedule, args.child_selection, args.seeds)
        for _, r in df_runs.iterrows():
            sig = r["signature"]
            exact_c = exact.get(sig, 0.0)
            err = abs(r["mean_concentration"] - exact_c)
            all_rows.append({
                "signature": sig,
                "q": q,
                "mean_concentration": r["mean_concentration"],
                "stdev_concentration": r["stdev_concentration"],
                "exact_concentration": exact_c,
                "abs_error": err,
            })
    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.output, index=False)
    print(f"Saved validation results to {args.output}")


if __name__ == "__main__":
    main()
