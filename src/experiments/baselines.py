from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import json

import pandas as pd
import networkx as nx

from src.algorithms.esa import (
    ESAParams,
    sample_many as esa_sample_many,
    sample_with_probabilities as esa_sample_with_probabilities,
    estimate_concentrations as esa_estimate_concentrations,
    parallel_esa_sample,
)
from src.utils.motifs import count_motif_signatures, canonical_signature
from src.utils.visualize import plot_motif_distribution, plot_motif_overlay


@dataclass
class ESABaselineResult:
    csv_path: Path
    meta_path: Path
    plot_path: Optional[Path]
    total_samples: int
    runtime_sec: float
    unique_classes: int


def _ensure_baseline_dir(output_dir: Path, dataset: str, k: int) -> Path:
    subdir = output_dir / dataset / f"k{k}" / "baseline" / "esa"
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def run_esa_baseline(
    G: nx.Graph,
    dataset: str,
    directed: bool,
    k: int,
    samples: int,
    seed: int,
    output_dir: Path,
    q: float,
    schedule: str,
    max_retries: int = 20,
    emit_plot: bool = True,
    compute_probabilities: bool = False,
    use_parallel: bool = False,
) -> ESABaselineResult:
    """Run the ESA baseline sampler for a fixed number of samples and persist results.

    Args:
        G: Graph or DiGraph to sample from (should already respect any node caps).
        dataset: Dataset key (used for output structure).
        directed: Whether the graph is directed (metadata only).
        k: Motif size.
        samples: Number of ESA samples to draw.
        seed: Random seed for reproducibility.
        output_dir: Root results directory.
        q: Sampling fraction comparable to RAND-ESU runs (stored in metadata for reference).
        schedule: RAND-ESU schedule name (stored for comparison).
        max_retries: Maximum retries inside ESA sampler.
        emit_plot: Whether to produce a simple concentration bar plot.
        compute_probabilities: If True, compute sampling probabilities and apply
            Equation (1) correction for unbiased concentration estimates.
            WARNING: This is O(k^k) per sample and very slow for large samples.
        use_parallel: If True, use parallel ESA sampling for speedup (ignored if
            compute_probabilities is True since that needs sequential probability tracking).

    Returns:
        ESABaselineResult describing saved artifacts.
    """
    if samples <= 0:
        raise ValueError("samples must be > 0 for ESA baseline")

    random.seed(seed)
    params = ESAParams(
        k=k, max_retries=max_retries, compute_probabilities=compute_probabilities
    )

    t0 = time.time()

    if compute_probabilities:
        # Use probability-corrected sampling (Equation 1 from article)
        esa_samples = list(esa_sample_with_probabilities(G, params, samples))
        total = len(esa_samples)

        # Define signature function for estimate_concentrations
        def sig_func(graph, vertices):
            subgraph = graph.subgraph(vertices)
            return canonical_signature(subgraph)

        # Get corrected concentrations using Equation (1)
        concentrations = esa_estimate_concentrations(
            G, esa_samples, sig_func, use_probability_correction=True
        )

        # Convert concentrations to frequency counts (for compatibility)
        # Note: these are "effective" counts based on probability weighting
        freq = {}
        for sig, conc in concentrations.items():
            freq[sig] = int(round(conc * total))  # Approximate count
    else:
        # Standard ESA without correction (biased but fast)
        if use_parallel:
            # Use parallel ESA sampling for speedup
            subgraphs = parallel_esa_sample(G, params, samples)
        else:
            subgraphs = list(esa_sample_many(G, params, samples))
        total, freq = count_motif_signatures(G, subgraphs)

    t1 = time.time()
    runtime = t1 - t0

    records = []
    for sig, count in freq.items():
        records.append(
            {
                "dataset": dataset,
                "directed": directed,
                "k": k,
                "q": q,
                "schedule": schedule,
                "seed": seed,
                "samples_requested": samples,
                "total_samples": total,
                "signature": sig,
                "count": count,
                "concentration": (count / total) if total > 0 else 0.0,
                "algo": "ESA",
            }
        )

    df = pd.DataFrame.from_records(records)

    subdir = _ensure_baseline_dir(output_dir, dataset, k)
    csv_path = subdir / f"esa_q{q}_seed{seed}_samples{samples}.csv"
    df.to_csv(csv_path, index=False)

    meta = {
        "dataset": dataset,
        "directed": directed,
        "k": k,
        "q": q,
        "schedule": schedule,
        "seed": seed,
        "samples_requested": samples,
        "total_samples": total,
        "unique_motif_classes": len(freq),
        "runtime_sec": runtime,
        "algo": "ESA",
        "max_retries": max_retries,
        "probability_correction": compute_probabilities,
    }
    meta_path = subdir / f"esa_q{q}_seed{seed}_samples{samples}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    plot_path: Optional[Path] = None
    if emit_plot and total > 0 and freq:
        plot_path = subdir / f"esa_q{q}_seed{seed}_samples{samples}_motifs.png"
        plot_motif_distribution(
            freq, total, f"ESA baseline {dataset} k={k} q={q} seed={seed}", plot_path
        )

    return ESABaselineResult(
        csv_path=csv_path,
        meta_path=meta_path,
        plot_path=plot_path,
        total_samples=total,
        runtime_sec=runtime,
        unique_classes=len(freq),
    )


def generate_overlay(
    dataset: str,
    k: int,
    q: float,
    seed: int,
    esa_freq: dict[str, int],
    esa_total: int,
    rand_freq: dict[str, int],
    rand_total: int,
    output_dir: Path,
) -> Path | None:
    """Produce ESA vs RAND-ESU overlay plot for qualitative bias visualization.

    Returns path of generated plot or None if failed.
    """
    try:
        if esa_total <= 0 or rand_total <= 0:
            return None
        subdir = output_dir / dataset / f"k{k}" / "overlay"

        subdir.mkdir(parents=True, exist_ok=True)
        out_path = subdir / f"overlay_k{k}_q{q}_seed{seed}.png"
        plot_motif_overlay(
            freq_a=esa_freq,
            total_a=esa_total,
            label_a="ESA",
            freq_b=rand_freq,
            total_b=rand_total,
            label_b="RAND-ESU",
            out_path=out_path,
            title=f"{dataset} k={k} q={q} seed={seed}: ESA vs RAND-ESU",
            top_n=20,
        )
        return out_path
    except Exception:
        return None


__all__ = [
    "run_esa_baseline",
    "ESABaselineResult",
    "generate_overlay",
]
