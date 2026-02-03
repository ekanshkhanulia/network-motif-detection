from __future__ import annotations

"""
Direct significance (k=3) using the EXACT Bender-Canfield method from Wernicke (2005) Section 3.

This implements Equation (3) from the article:

⟨Ĉ^i_k(G)⟩ = Σ_{v1,...,vk} |{G' ∈ DegSeq(G) | G'[{v1,...,vk}] ∈ S^i_k}|
             ──────────────────────────────────────────────────────────────
             Σ_{v1,...,vk} |{G' ∈ DegSeq(G) | G'[{v1,...,vk}] connected}|

The Bender-Canfield theorems allow computing the number of graphs with a fixed subgraph:
- Bender (1974): For directed graphs (non-negative matrices with row/column sums)
- Bender & Canfield (1978): For undirected graphs (labeled graphs with degree sequences)

For the ratio N(d_reduced)/N(d_original), we use an INCREMENTAL formula that only
depends on the degrees of the k=3 vertices and the pattern edges, making it O(1)
per pattern instead of O(n).

LAMBDA CORRECTION:
The full Bender-Canfield formulas include a λ correction term:
- Undirected: exp(-λ/2 - λ²/4) where λ = Σd_i²/2m
- Directed: exp(-λ) where λ = Σ(d_out_i · d_in_i)/m

When computing ratios N(d_reduced)/N(d_original), the λ terms contribute:
- Undirected: -(λ' - λ)/2 - (λ'² - λ²)/4
- Directed: -(λ' - λ)

By default, λ correction is disabled (asymptotic approximation valid for large n >> k).
Enable with include_lambda_correction=True for higher precision.
"""

import argparse
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import networkx as nx
import pandas as pd

from src.config import DATASETS, resolve_data_path
from src.utils.io import load_snap_graph
from src.utils.motifs import (
    canonical_signature,
    induced_subgraph_nodes,
    triad_label_from_signature,
)
from src.algorithms.rand_esu import parallel_esu_enumerate_with_signatures


# ============================================================================
# Lambda Computation for Bender-Canfield Correction
# ============================================================================


def _compute_lambda_undirected(degrees: Dict[int, int], m: int) -> float:
    """Compute λ = Σd_i² / (2m) for undirected graphs.

    This is the expected number of multiple edges in a random graph with
    the given degree sequence.
    """
    if m == 0:
        return 0.0
    sum_d_squared = sum(d * d for d in degrees.values())
    return sum_d_squared / (2 * m)


def _compute_lambda_directed(
    in_degrees: Dict[int, int], out_degrees: Dict[int, int], m: int
) -> float:
    """Compute λ = Σ(d_out_i · d_in_i) / m for directed graphs.

    This is the expected number of self-loops in a random directed graph
    with the given degree sequence.
    """
    if m == 0:
        return 0.0
    sum_product = sum(
        out_degrees.get(v, 0) * in_degrees.get(v, 0)
        for v in set(out_degrees.keys()) | set(in_degrees.keys())
    )
    return sum_product / m


# ============================================================================
# Bender-Canfield Incremental Ratio Computation
# ============================================================================


def _log_factorial(n: int) -> float:
    """Compute log(n!) using lgamma."""
    if n <= 1:
        return 0.0
    return math.lgamma(n + 1)


def _bc_log_ratio_undirected(
    degrees: Dict[int, int],
    nodes: Tuple[int, int, int],
    edges: List[Tuple[int, int]],
    m: int,
    include_lambda: bool = False,
    lambda_original: Optional[float] = None,
) -> float:
    """
    Compute log(N(d_reduced) / N(d_original)) for undirected graphs.

    Using the Bender-Canfield (1978) formula:
        N(d) ≈ (Σd_i)! / (∏d_i! · 2^m · m!) · exp(-λ/2 - λ²/4)

    When we fix k edges, we reduce m to m' = m - k, and reduce degrees d_i to d_i'.

    log(N_red/N_orig) = log((Σd')!) - log((Σd)!)           [numerator change]
                      - Σ[log(d_i'!) - log(d_i!)]         [degree factorial change]
                      - (m' - m)·log(2)                    [2^m change: negative since m'<m]
                      - [log(m'!) - log(m!)]              [m! change: negative since m'<m]
                      - (λ' - λ)/2 - (λ'² - λ²)/4         [λ correction, optional]

    Since Σd' = Σd - 2k (each edge removes 2 from sum of degrees):
        log((Σd')!) - log((Σd)!) = -Σ_{i=0}^{2k-1} log(Σd - i)
    """
    u, v, w = nodes
    node_map = {0: u, 1: v, 2: w}

    # Count edges incident to each of the 3 nodes
    edge_count = {u: 0, v: 0, w: 0}
    for e in edges:
        n1, n2 = node_map[e[0]], node_map[e[1]]
        edge_count[n1] = edge_count.get(n1, 0) + 1
        edge_count[n2] = edge_count.get(n2, 0) + 1

    num_edges = len(edges)
    m_prime = m - num_edges

    if m_prime < 0:
        return float("-inf")

    # Original degrees
    d_u, d_v, d_w = degrees.get(u, 0), degrees.get(v, 0), degrees.get(w, 0)

    # Reduced degrees
    d_u_prime = d_u - edge_count.get(u, 0)
    d_v_prime = d_v - edge_count.get(v, 0)
    d_w_prime = d_w - edge_count.get(w, 0)

    # Check validity
    if d_u_prime < 0 or d_v_prime < 0 or d_w_prime < 0:
        return float("-inf")

    # Compute log ratio
    log_ratio = 0.0

    # Change in (Σd)!: log((2m')!) - log((2m)!) = log((2m-2k)!) - log((2m)!)
    # = -Σ_{i=0}^{2k-1} log(2m - i)
    sum_d = 2 * m
    for i in range(2 * num_edges):
        if sum_d - i > 0:
            log_ratio -= math.log(sum_d - i)
        else:
            return float("-inf")

    # Change in degree factorials: -Σ[log(d_i'!) - log(d_i!)] = Σ[log(d_i!) - log(d_i'!)]
    # This is POSITIVE because we're dividing by SMALLER factorials
    log_ratio += _log_factorial(d_u) - _log_factorial(d_u_prime)
    log_ratio += _log_factorial(d_v) - _log_factorial(d_v_prime)
    log_ratio += _log_factorial(d_w) - _log_factorial(d_w_prime)

    # Change in 2^m: (m' - m)·log(2) = -k·log(2) [ADDS to ratio because denominator shrinks]
    log_ratio += num_edges * math.log(2)

    # Change in m!: log(m'!) - log(m!) = -Σ_{i=0}^{k-1} log(m - i) [SUBTRACTS]
    for i in range(num_edges):
        if m - i > 0:
            log_ratio -= math.log(m - i)
        else:
            return float("-inf")

    # Lambda correction (optional)
    if include_lambda and lambda_original is not None:
        # Compute λ' for reduced degree sequence
        # Only the 3 nodes change: Σd'² = Σd² - (d_u² + d_v² + d_w²) + (d_u'² + d_v'² + d_w'²)
        delta_sum_d_sq = (
            (d_u_prime * d_u_prime - d_u * d_u)
            + (d_v_prime * d_v_prime - d_v * d_v)
            + (d_w_prime * d_w_prime - d_w * d_w)
        )

        # λ = Σd²/(2m), λ' = (Σd² + delta)/(2m')
        # For incremental: λ' ≈ (λ * 2m + delta) / (2m')
        if m_prime > 0:
            lambda_prime = (lambda_original * 2 * m + delta_sum_d_sq) / (2 * m_prime)
        else:
            lambda_prime = 0.0

        # Correction: -(λ' - λ)/2 - (λ'² - λ²)/4
        lambda_diff = lambda_prime - lambda_original
        lambda_sq_diff = lambda_prime * lambda_prime - lambda_original * lambda_original
        log_ratio -= lambda_diff / 2 + lambda_sq_diff / 4

    return log_ratio


def _bc_log_ratio_directed(
    in_degrees: Dict[int, int],
    out_degrees: Dict[int, int],
    nodes: Tuple[int, int, int],
    edges: List[Tuple[int, int]],
    m: int,
    include_lambda: bool = False,
    lambda_original: Optional[float] = None,
) -> float:
    """
    Compute log(N(d_reduced) / N(d_original)) for directed graphs.

    Using Bender (1974) formula for directed graphs:
        N(d_in, d_out) ≈ m! / (∏d_out_i! · ∏d_in_j!) · exp(-λ)

    where λ = Σ(d_out_i · d_in_i) / m

    When we fix k edges:
        log(N_red/N_orig) = log(m'!/m!)                           [NEGATIVE: m' < m]
                          + Σ[log(d_out_i!) - log(d_out_i'!)]     [POSITIVE: d' < d]
                          + Σ[log(d_in_i!) - log(d_in_i'!)]       [POSITIVE: d' < d]
                          - (λ' - λ)                               [λ correction, optional]
    """
    u, v, w = nodes
    node_map = {0: u, 1: v, 2: w}

    # Count in/out edges for each node
    out_count = {u: 0, v: 0, w: 0}
    in_count = {u: 0, v: 0, w: 0}

    for e in edges:
        src, dst = node_map[e[0]], node_map[e[1]]
        out_count[src] = out_count.get(src, 0) + 1
        in_count[dst] = in_count.get(dst, 0) + 1

    num_edges = len(edges)
    m_prime = m - num_edges

    if m_prime < 0:
        return float("-inf")

    # Original degrees
    d_out_u = out_degrees.get(u, 0)
    d_out_v = out_degrees.get(v, 0)
    d_out_w = out_degrees.get(w, 0)
    d_in_u = in_degrees.get(u, 0)
    d_in_v = in_degrees.get(v, 0)
    d_in_w = in_degrees.get(w, 0)

    # Reduced degrees
    d_out_u_prime = d_out_u - out_count.get(u, 0)
    d_out_v_prime = d_out_v - out_count.get(v, 0)
    d_out_w_prime = d_out_w - out_count.get(w, 0)
    d_in_u_prime = d_in_u - in_count.get(u, 0)
    d_in_v_prime = d_in_v - in_count.get(v, 0)
    d_in_w_prime = d_in_w - in_count.get(w, 0)

    # Check validity
    if any(
        d < 0
        for d in [
            d_out_u_prime,
            d_out_v_prime,
            d_out_w_prime,
            d_in_u_prime,
            d_in_v_prime,
            d_in_w_prime,
        ]
    ):
        return float("-inf")

    # Compute log ratio
    log_ratio = 0.0

    # Change in m!: log(m'!) - log(m!) = -Σ_{i=0}^{k-1} log(m - i) [NEGATIVE]
    for i in range(num_edges):
        if m - i > 0:
            log_ratio -= math.log(m - i)
        else:
            return float("-inf")

    # Change in out-degree factorials (in denominator, so sign flips):
    # We divide by SMALLER factorials in reduced, so ratio is LARGER
    log_ratio += _log_factorial(d_out_u) - _log_factorial(d_out_u_prime)
    log_ratio += _log_factorial(d_out_v) - _log_factorial(d_out_v_prime)
    log_ratio += _log_factorial(d_out_w) - _log_factorial(d_out_w_prime)

    # Change in in-degree factorials (same logic)
    log_ratio += _log_factorial(d_in_u) - _log_factorial(d_in_u_prime)
    log_ratio += _log_factorial(d_in_v) - _log_factorial(d_in_v_prime)
    log_ratio += _log_factorial(d_in_w) - _log_factorial(d_in_w_prime)

    # Lambda correction (optional)
    if include_lambda and lambda_original is not None:
        # Compute change in λ = Σ(d_out · d_in) / m
        # Only the 3 nodes change their products
        delta_product = (
            (d_out_u_prime * d_in_u_prime - d_out_u * d_in_u)
            + (d_out_v_prime * d_in_v_prime - d_out_v * d_in_v)
            + (d_out_w_prime * d_in_w_prime - d_out_w * d_in_w)
        )

        # λ' = (λ * m + delta) / m'
        if m_prime > 0:
            lambda_prime = (lambda_original * m + delta_product) / m_prime
        else:
            lambda_prime = 0.0

        # Correction: -(λ' - λ)
        log_ratio -= lambda_prime - lambda_original

    return log_ratio


# ============================================================================
# Pattern Generation
# ============================================================================


def _get_all_connected_triads_directed() -> List[Tuple[str, List[Tuple[int, int]]]]:
    """
    Generate all connected directed triad patterns with their edge lists.
    Returns list of (signature, edge_list) tuples.
    """
    triads = []
    all_edges = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]

    for mask in range(64):
        edges = [e for i, e in enumerate(all_edges) if mask & (1 << i)]

        G = nx.DiGraph()
        G.add_nodes_from([0, 1, 2])
        G.add_edges_from(edges)

        # Check weak connectivity
        if len(edges) >= 2:  # Need at least 2 edges for connectivity in 3 nodes
            U = G.to_undirected()
            if U.number_of_edges() >= 2 and nx.is_connected(U):
                sig = canonical_signature(G)
                triads.append((sig, edges))

    return triads


def _get_all_connected_triads_undirected() -> List[Tuple[str, List[Tuple[int, int]]]]:
    """
    Generate all connected undirected triad patterns.
    Returns list of (signature, edge_list) tuples.
    """
    triads = []
    all_edges = [(0, 1), (0, 2), (1, 2)]

    for mask in range(8):
        edges = [e for i, e in enumerate(all_edges) if mask & (1 << i)]

        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        G.add_edges_from(edges)

        if len(edges) >= 2 and nx.is_connected(G):
            sig = canonical_signature(G)
            triads.append((sig, edges))

    return triads


# ============================================================================
# Sampling
# ============================================================================


def _sample_triples(n: int, T: int, seed: int = 1) -> Iterable[Tuple[int, int, int]]:
    """Sample T unique vertex triples uniformly at random."""
    rng = random.Random(seed)
    seen = set()
    attempts = 0
    max_attempts = T * 100

    while len(seen) < T and attempts < max_attempts:
        u, v, w = rng.sample(range(n), 3)
        key = tuple(sorted((u, v, w)))
        if key not in seen:
            seen.add(key)
            yield (u, v, w)
        attempts += 1


def _sample_triples_list(n: int, T: int, seed: int = 1) -> List[Tuple[int, int, int]]:
    """Sample T unique vertex triples uniformly at random, returning a list."""
    return list(_sample_triples(n, T, seed))


# ============================================================================
# Parallel Processing Workers
# ============================================================================


def _process_triple_batch(
    batch: List[Tuple[int, int, int]],
    G_data: dict,
    all_patterns: List[Tuple[str, List[Tuple[int, int]]]],
    is_directed: bool,
    degrees: Dict[int, int],
    in_degrees: Dict[int, int],
    out_degrees: Dict[int, int],
    m: int,
    include_lambda_correction: bool,
    lambda_original: Optional[float],
) -> Tuple[Dict[str, float], float, Dict[str, int], int]:
    """
    Process a batch of vertex triples for BC computation.

    Returns:
        (exp_num, exp_den, obs_counts, obs_connected) for this batch
    """
    # Reconstruct graph from edge list
    if is_directed:
        G = nx.DiGraph()
        G.add_edges_from(G_data["edges"])
    else:
        G = nx.Graph()
        G.add_edges_from(G_data["edges"])

    exp_num: Dict[str, float] = {}
    exp_den = 0.0
    obs_counts: Dict[str, int] = {}
    obs_connected = 0

    for u, v, w in batch:
        nodes = (u, v, w)

        # Compute observed pattern in original graph
        sub = induced_subgraph_nodes(G, [u, v, w])
        obs_sig = canonical_signature(sub)

        # Check if connected
        if is_directed:
            U = sub.to_undirected()
            is_conn = U.number_of_edges() >= 2 and nx.is_connected(U)
        else:
            is_conn = sub.number_of_edges() >= 2 and nx.is_connected(sub)

        if is_conn:
            obs_counts[obs_sig] = obs_counts.get(obs_sig, 0) + 1
            obs_connected += 1

        # Compute BC ratio for each connected pattern at this triple
        for sig, edges in all_patterns:
            if is_directed:
                log_ratio = _bc_log_ratio_directed(
                    in_degrees,
                    out_degrees,
                    nodes,
                    edges,
                    m,
                    include_lambda=include_lambda_correction,
                    lambda_original=lambda_original,
                )
            else:
                log_ratio = _bc_log_ratio_undirected(
                    degrees,
                    nodes,
                    edges,
                    m,
                    include_lambda=include_lambda_correction,
                    lambda_original=lambda_original,
                )

            # Convert log ratio to probability weight
            if log_ratio > -700:
                p = math.exp(log_ratio)
            else:
                p = 0.0

            exp_num[sig] = exp_num.get(sig, 0.0) + p
            exp_den += p

    return exp_num, exp_den, obs_counts, obs_connected


# ============================================================================
# Main Algorithm
# ============================================================================


def run_direct_significance_bc(
    dataset: str,
    data_dir: Path,
    output_dir: Path,
    T: int = 100000,
    seed: int = 1,
    max_nodes: int | None = None,
    include_lambda_correction: bool = False,
    parallel: bool = True,
    max_workers: int | None = None,
) -> Dict[str, object]:
    """
    Direct significance for triads (k=3) using exact Bender-Canfield method.

    Implements Wernicke (2005) Section 3, Equation (3):
    - Sample T random vertex triples {v1, v2, v3}
    - For each triple, compute probability ratio for each connected pattern
    - Accumulate numerators (per pattern) and denominator (any connected)
    - Expected concentration = Σ P(pattern) / Σ P(connected)

    Parameters match article Table 2: T=100,000 samples.

    Uses OPTIMIZED incremental BC computation - O(1) per pattern instead of O(n).

    Args:
        dataset: Name of dataset to process
        data_dir: Directory containing data files
        output_dir: Directory for output files
        T: Number of random vertex triples to sample (article: 100,000)
        seed: Random seed for reproducibility
        max_nodes: Optional limit on number of nodes to load
        include_lambda_correction: If True, include λ correction term in BC formula
            (more accurate but slightly slower). Default False uses asymptotic
            approximation valid for large graphs.
        parallel: If True, use parallel processing across CPU cores
        max_workers: Number of worker processes (defaults to cpu_count - 2)
    """
    random.seed(seed)

    cfg = DATASETS.get(dataset)
    if cfg is None:
        raise SystemExit(f"Unknown dataset {dataset}")

    path = resolve_data_path(data_dir, dataset)
    G = load_snap_graph(path, directed=cfg.directed, max_nodes=max_nodes)

    n = G.number_of_nodes()
    m = G.number_of_edges()
    is_directed = G.is_directed()

    print(f"Dataset: {dataset}")
    print(f"  Nodes: {n}, Edges: {m}, Directed: {is_directed}")
    print(f"  Sampling T={T} vertex triples...")
    print(
        f"  Lambda correction: {'ENABLED' if include_lambda_correction else 'DISABLED (asymptotic approx)'}"
    )

    # Initialize degree dictionaries
    in_degrees: Dict[int, int] = {}
    out_degrees: Dict[int, int] = {}
    degrees: Dict[int, int] = {}
    lambda_original: Optional[float] = None

    # Precompute degree sequences
    if is_directed:
        for node in G.nodes():
            in_degrees[node] = len(list(G.predecessors(node)))  # type: ignore
            out_degrees[node] = len(list(G.successors(node)))  # type: ignore
        all_patterns = _get_all_connected_triads_directed()

        # Compute original λ if correction is enabled
        if include_lambda_correction:
            lambda_original = _compute_lambda_directed(in_degrees, out_degrees, m)
            print(f"  Original λ (directed): {lambda_original:.6f}")
    else:
        for node in G.nodes():
            degrees[node] = len(list(G.neighbors(node)))
        all_patterns = _get_all_connected_triads_undirected()

        # Compute original λ if correction is enabled
        if include_lambda_correction:
            lambda_original = _compute_lambda_undirected(degrees, m)
            print(f"  Original λ (undirected): {lambda_original:.6f}")

    print(
        f"  Found {len(all_patterns)} connected triad patterns (including permutations)"
    )

    # Accumulators
    exp_num: Dict[str, float] = {}  # Expected numerator per signature
    exp_den = 0.0  # Expected denominator (sum over all connected)

    # Observed counts for reference (populated after expected calculations)
    obs_counts: Dict[str, int] = {}
    obs_connected = 0
    observed_source = "sampled_triples"

    # Sample all vertex triples upfront
    all_triples = _sample_triples_list(n, T, seed)
    sampled = len(all_triples)
    print(f"  Sampled {sampled} unique triples")

    # Prepare graph data for serialization to workers
    G_data = {"edges": list(G.edges())}

    if parallel and sampled > 1000:
        # Parallel processing
        cpu_cnt = os.cpu_count() or 4
        num_workers = max_workers if max_workers else max(1, cpu_cnt - 2)
        batch_size = max(1000, sampled // (num_workers * 4))  # At least 1000 per batch
        batches = [
            all_triples[i : i + batch_size] for i in range(0, sampled, batch_size)
        ]

        print(
            f"  Using {num_workers} workers, {len(batches)} batches of ~{batch_size} triples each"
        )

        completed_batches = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _process_triple_batch,
                    batch,
                    G_data,
                    all_patterns,
                    is_directed,
                    degrees,
                    in_degrees,
                    out_degrees,
                    m,
                    include_lambda_correction,
                    lambda_original,
                )
                for batch in batches
            ]

            for future in as_completed(futures):
                batch_exp_num, batch_exp_den, batch_obs_counts, batch_obs_connected = (
                    future.result()
                )

                # Merge results
                for sig, val in batch_exp_num.items():
                    exp_num[sig] = exp_num.get(sig, 0.0) + val
                exp_den += batch_exp_den

                for sig, count in batch_obs_counts.items():
                    obs_counts[sig] = obs_counts.get(sig, 0) + count
                obs_connected += batch_obs_connected

                completed_batches += 1
                if completed_batches % max(1, len(batches) // 10) == 0:
                    print(f"  Completed {completed_batches}/{len(batches)} batches...")

    else:
        # Sequential processing (fallback)
        print("  Using sequential processing")
        for i, (u, v, w) in enumerate(all_triples):
            nodes = (u, v, w)

            # Compute observed pattern in original graph
            sub = induced_subgraph_nodes(G, [u, v, w])
            obs_sig = canonical_signature(sub)

            # Check if connected
            if is_directed:
                U = sub.to_undirected()
                is_conn = U.number_of_edges() >= 2 and nx.is_connected(U)
            else:
                is_conn = sub.number_of_edges() >= 2 and nx.is_connected(sub)

            if is_conn:
                obs_counts[obs_sig] = obs_counts.get(obs_sig, 0) + 1
                obs_connected += 1

            # Compute BC ratio for each connected pattern at this triple
            for sig, edges in all_patterns:
                if is_directed:
                    log_ratio = _bc_log_ratio_directed(
                        in_degrees,
                        out_degrees,
                        nodes,
                        edges,
                        m,
                        include_lambda=include_lambda_correction,
                        lambda_original=lambda_original,
                    )
                else:
                    log_ratio = _bc_log_ratio_undirected(
                        degrees,
                        nodes,
                        edges,
                        m,
                        include_lambda=include_lambda_correction,
                        lambda_original=lambda_original,
                    )

                # Convert log ratio to probability weight
                if log_ratio > -700:
                    p = math.exp(log_ratio)
                else:
                    p = 0.0

                exp_num[sig] = exp_num.get(sig, 0.0) + p
                exp_den += p

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/{sampled} triples...")

    print(f"  Completed {sampled} triples")
    try:
        print("  Enumerating observed triads in original graph (full enumeration)...")
        obs_connected, obs_counts = parallel_esu_enumerate_with_signatures(G, 3)
        observed_source = "full_enumeration"
        print(f"  Observed {obs_connected} connected triads (full enumeration)")
    except Exception as exc:
        print(
            f"  [WARN] Enumeration failed ({exc}); using sampled triples for observed counts"
        )
        print(f"  Observed {obs_connected} connected triads (sampled)")

    # Compute results
    rows = []
    all_sigs = set(list(exp_num.keys()) + list(obs_counts.keys()))

    for sig in sorted(all_sigs):
        # Expected concentration in random ensemble
        exp_c = (exp_num.get(sig, 0.0) / exp_den) if exp_den > 0 else 0.0

        # Observed concentration in original graph
        obs_c = (obs_counts.get(sig, 0) / obs_connected) if obs_connected > 0 else 0.0

        # Z-score
        var = exp_c * (1 - exp_c) / obs_connected if obs_connected > 0 else 0.0
        z = (obs_c - exp_c) / math.sqrt(var) if var > 0 else 0.0
        if exp_c > 0:
            ratio = obs_c / exp_c
        elif obs_c > 0:
            ratio = float("inf")
        else:
            ratio = float("nan")

        rows.append(
            {
                "dataset": dataset,
                "signature": sig,
                "triad": triad_label_from_signature(sig)
                if is_directed
                else ("triangle" if "111" in sig else "path"),
                "expected_concentration": exp_c,
                "observed_concentration": obs_c,
                "enrichment_ratio": ratio,
                "z_score": z,
                "T": T,
                "T_connected": obs_connected,
                "n": n,
                "m": m,
                "seed": seed,
                "method": "exact_bender_canfield",
                "lambda_correction": include_lambda_correction,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("observed_concentration", ascending=False)

    # Save results
    lambda_suffix = "_with_lambda" if include_lambda_correction else ""
    out_dir = output_dir / dataset / "k3" / "significance_direct_bc"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / f"direct_bc_exact_T{T}_seed{seed}{lambda_suffix}.csv"
    df.to_csv(out_csv, index=False)

    meta = {
        "dataset": dataset,
        "T": T,
        "T_connected": obs_connected,
        "observed_source": observed_source,
        "seed": seed,
        "n": n,
        "m": m,
        "directed": is_directed,
        "method": "exact_bender_canfield",
        "lambda_correction": include_lambda_correction,
        "lambda_original": lambda_original,
        "description": "Exact Bender-Canfield method per Wernicke (2005) Section 3",
        "ratio_definition": "observed_concentration / expected_concentration",
        "observed_note": (
            "Observed concentrations come from full enumeration of connected triads "
            "in the original graph (ESU), not from sampled triples."
        ),
        "z_score_note": (
            "Z-scores are included for reference; Wernicke (2005) reports ratios "
            "of observed to expected concentrations. Z can be extreme when expected "
            "variance is tiny."
        ),
    }
    meta_path = out_dir / f"direct_bc_exact_T{T}_seed{seed}{lambda_suffix}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nResults saved to: {out_csv}")
    print("\nTop 5 most frequent triads (observed):")
    for _, row in df.head(5).iterrows():
        print(
            f"  {row['signature']} ({row['triad']}): "
            f"obs={row['observed_concentration']:.4f}, "
            f"exp={row['expected_concentration']:.4f}, "
            f"ratio={row['enrichment_ratio']:.3f}, "
            f"Z={row['z_score']:.1f}"
        )

    return {"df": df, "csv_path": out_csv, "meta_path": meta_path, "subdir": out_dir}


def main():
    parser = argparse.ArgumentParser(
        description="Direct significance using exact Bender-Canfield method (Wernicke 2005 Section 3).\n"
        "Computes expected motif concentrations in graphs with same degree sequence.\n"
        "Article Table 2 used T=100,000 samples."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--T",
        type=int,
        default=100000,
        help="Number of random vertex triples to sample (article: 100000)",
    )
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--include-lambda-correction",
        action="store_true",
        help="Include λ correction term in BC formula (more accurate, slightly slower)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker processes (default: cpu_count - 2)",
    )
    args = parser.parse_args()

    run_direct_significance_bc(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        T=args.T,
        seed=args.seed,
        max_nodes=args.max_nodes,
        include_lambda_correction=args.include_lambda_correction,
        parallel=not args.no_parallel,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
