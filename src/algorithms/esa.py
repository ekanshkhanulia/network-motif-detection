"""ESA (Edge Sampling Algorithm) baseline per Kashtan et al. (Bioinformatics 2004).

This implementation follows the original ESA algorithm as described in:
- Kashtan et al. (2004) "Efficient sampling algorithm for estimating subgraph
  concentrations and detecting network motifs"
- Wernicke (2005) Section 2.1 "The Previous Approach: Edge Sampling"

IMPORTANT NOTE ON DIRECTED GRAPHS:
Per the original Wernicke (2005) paper's recommendation, when sampling from
directed graphs, ESA treats the network as UNDIRECTED for the purpose of
subgraph expansion (using "weak connectivity" - following edges in both
directions). The resulting k-vertex subgraph is then analyzed with its
original directed edges for motif classification.

PROBABILITY CORRECTION (Equation 1):
ESA has inherent sampling bias - some subgraphs are sampled more frequently
than others. To obtain unbiased concentration estimates, the article provides
Equation (1):

    C^i_k(R,G) = sum_{G' in R, G' in S^i_k} (1/Pr[G' sampled])
                 -------------------------------------------------
                 sum_{G' in R} (1/Pr[G' sampled])

Computing Pr[G' sampled] requires enumerating all paths ESA could take to
reach that subgraph, which is O(k^k) per sample. This is expensive but
necessary for unbiased estimation.

This implementation provides:
1. naive_edge_sampling(): Returns (vertex_set, expansion_history) for one sample
2. compute_sampling_probability(): Computes Pr[subgraph sampled] per Equation (1)
3. sample_with_probabilities(): Samples with probability computation
4. estimate_concentrations(): Computes unbiased concentration estimates

The probability correction can be disabled for speed-only benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence, Set, Optional, Dict, List, Tuple
from collections import defaultdict
import random
import networkx as nx

__all__ = [
    "ESAParams",
    "ESASample",
    "naive_edge_sampling",
    "sample_many",
    "parallel_esa_sample",
    "compute_sampling_probability",
    "sample_with_probabilities",
    "estimate_concentrations",
]


@dataclass
class ESAParams:
    k: int
    max_retries: int = 20  # retries if expansion fails early
    compute_probabilities: bool = False  # Enable probability correction (Equation 1)


@dataclass
class ESASample:
    """Result of an ESA sample with optional probability information."""

    vertices: Tuple[int, ...]
    initial_edge: Tuple[int, int]
    expansion_history: List[Tuple[int, int]]  # List of (from_vertex, new_vertex) pairs
    sampling_probability: Optional[float] = None


def _random_edge(G: nx.Graph):
    """Return a random edge as a tuple (u,v)."""
    edges = getattr(G, "_cached_edge_list", None)
    if edges is None:
        edges = list(G.edges())
        G._cached_edge_list = edges  # type: ignore[attr-defined]
    if not edges:
        raise ValueError("Graph has no edges")
    return random.choice(edges)


def _get_neighbors(G, node: int) -> set:
    """Get all neighbors of a node, treating directed graphs as undirected.

    For directed graphs, returns both predecessors and successors (weak connectivity).
    This is the correct behavior per Wernicke (2005) - ESA samples using undirected
    connectivity, then the resulting subgraph is analyzed with original directions.
    """
    if G.is_directed():
        return set(G.predecessors(node)) | set(G.successors(node))
    else:
        return set(G.neighbors(node))


def _get_effective_degree(G, vertex: int, current_set: Set[int]) -> int:
    """Compute effective degree of vertex w.r.t. current vertex set.

    Effective degree = total degree - edges to vertices already in the set.
    This is used in probability computation per mfinder implementation.
    """
    neighbors = _get_neighbors(G, vertex)
    # Count neighbors not in current set
    return len(neighbors - current_set)


def naive_edge_sampling(G: nx.Graph, params: ESAParams) -> Optional[ESASample]:
    """Perform one ESA sample returning an ESASample or None on failure.

    Algorithm (Kashtan et al. 2004):
    1. Pick a random edge {u,v}.
    2. Initialize V' = {u,v}.
    3. While |V'| < k:
         - Consider edges in V' x N(V') (neighbors of current set) and pick
           one uniformly at random; add any new endpoint(s) to V'.
         - If no such edge adds a new vertex, restart (retry) until retry cap.
    4. Return the k-set with expansion history.
    """
    k = params.k
    if k < 2:
        raise ValueError("k must be >= 2 for ESA")
    if G.number_of_nodes() < k:
        return None

    retries = 0
    while retries <= params.max_retries:
        try:
            initial_edge = _random_edge(G)
            u, v = initial_edge
        except ValueError:
            return None

        sub: Set[int] = {u, v}
        expansion_history: List[Tuple[int, int]] = []

        if len(sub) < 2:  # self-loop scenario
            retries += 1
            continue

        # Expand
        stalled = False
        while len(sub) < k:
            frontier_edges = []
            for x in list(sub):
                for y in _get_neighbors(G, x):
                    if y not in sub:
                        frontier_edges.append((x, y))

            if not frontier_edges:
                stalled = True
                break

            chosen_edge = random.choice(frontier_edges)
            from_vertex, new_vertex = chosen_edge
            sub.add(new_vertex)
            expansion_history.append((from_vertex, new_vertex))

        if stalled:
            retries += 1
            continue

        if len(sub) == k:
            return ESASample(
                vertices=tuple(sub),
                initial_edge=initial_edge,
                expansion_history=expansion_history,
                sampling_probability=None,
            )
        retries += 1

    return None


def compute_sampling_probability(G: nx.Graph, vertices: Sequence[int], k: int) -> float:
    """Compute Pr[subgraph sampled by ESA] for probability correction (Equation 1).

    This is a recursive computation that considers all possible orderings
    in which ESA could have reached this subgraph. The complexity is O(k^k)
    in the worst case.

    The probability is:
    Pr[subgraph] = sum over all starting edges e in subgraph:
                     (1/|E|) * Pr[reach full subgraph | started with e]

    Where Pr[reach | started with e] is computed recursively by considering
    all possible expansion paths.

    Per mfinder implementation (prob.c, get_p_i_efc function).
    """
    vertex_set = set(vertices)
    n_edges = G.number_of_edges()

    if n_edges == 0:
        return 0.0

    if len(vertex_set) == 2:
        # Base case: probability of picking this exact edge
        return 1.0 / n_edges

    # Build induced subgraph to find internal edges
    subgraph = G.subgraph(vertex_set)
    if G.is_directed():
        # For directed graphs, we need edges in both directions
        internal_edges = list(subgraph.edges())
    else:
        internal_edges = list(subgraph.edges())

    if not internal_edges:
        return 0.0

    total_prob = 0.0

    # For each possible starting edge in the subgraph
    # Note: For directed graphs, internal_edges already includes both directions
    # when mutual edges exist, so we should not add an extra reverse case here.
    for start_edge in internal_edges:
        s, t = start_edge
        # Probability of starting with this edge
        p_start = 1.0 / n_edges

        # Compute probability of reaching full subgraph from this start
        p_reach = _compute_reach_probability(G, vertex_set, {s, t}, k)

        total_prob += p_start * p_reach

    return total_prob


def _compute_reach_probability(
    G: nx.Graph, target_set: Set[int], current_set: Set[int], k: int
) -> float:
    """Recursively compute probability of reaching target_set from current_set.

    This implements the recursive probability computation from mfinder's
    get_p_i_ function.
    """
    if current_set == target_set:
        return 1.0

    if len(current_set) >= k:
        return 0.0

    # Find vertices in target but not in current
    remaining = target_set - current_set

    if not remaining:
        return 1.0

    # Compute frontier: edges from current_set to neighbors not in current_set
    frontier_edges = []
    for x in current_set:
        for y in _get_neighbors(G, x):
            if y not in current_set:
                frontier_edges.append((x, y))

    if not frontier_edges:
        return 0.0

    # Denominator: total number of frontier edges
    total_frontier = len(frontier_edges)

    # Sum probability over all vertices in remaining that we could add next
    total_prob = 0.0

    for v in remaining:
        # Count edges from current_set to v (numerator)
        edges_to_v = sum(1 for x in current_set if v in _get_neighbors(G, x))

        if edges_to_v == 0:
            continue

        # Probability of choosing an edge to v
        p_choose_v = edges_to_v / total_frontier

        # Recursively compute probability of completing from current_set + {v}
        new_set = current_set | {v}
        p_complete = _compute_reach_probability(G, target_set, new_set, k)

        total_prob += p_choose_v * p_complete

    return total_prob


def sample_with_probabilities(
    G: nx.Graph, params: ESAParams, n_samples: int
) -> Iterator[ESASample]:
    """Generate ESA samples with probability computation.

    If params.compute_probabilities is True, each sample includes its
    sampling probability for use in Equation (1) correction.
    """
    produced = 0
    while produced < n_samples:
        sample = naive_edge_sampling(G, params)
        if sample is not None:
            if params.compute_probabilities:
                sample.sampling_probability = compute_sampling_probability(
                    G, sample.vertices, params.k
                )
            produced += 1
            yield sample


def sample_many(
    G: nx.Graph, params: ESAParams, n_samples: int
) -> Iterator[Sequence[int]]:
    """Generate up to n_samples ESA k-subgraphs (vertex tuples).

    This is the simple interface that just returns vertex tuples.
    For probability-corrected sampling, use sample_with_probabilities().
    """
    produced = 0
    while produced < n_samples:
        sample = naive_edge_sampling(G, params)
        if sample is not None:
            produced += 1
            yield sample.vertices


def estimate_concentrations(
    G: nx.Graph,
    samples: List[ESASample],
    signature_func,
    use_probability_correction: bool = True,
) -> Dict[str, float]:
    """Estimate subgraph class concentrations from ESA samples.

    If use_probability_correction is True, applies Equation (1) from the article:

        C^i_k(R,G) = sum_{G' in R, G' in S^i_k} (1/Pr[G' sampled])
                     -------------------------------------------------
                     sum_{G' in R} (1/Pr[G' sampled])

    Otherwise, returns naive frequency estimates (biased).

    Args:
        G: The graph being sampled
        samples: List of ESASample objects (with probabilities if correction enabled)
        signature_func: Function that takes (G, vertices) and returns canonical signature
        use_probability_correction: Whether to apply Equation (1) correction

    Returns:
        Dictionary mapping signature -> estimated concentration
    """
    if not samples:
        return {}

    # Group samples by signature
    sig_weights: Dict[str, float] = defaultdict(float)
    total_weight = 0.0

    for sample in samples:
        sig = signature_func(G, sample.vertices)

        if use_probability_correction and sample.sampling_probability is not None:
            if sample.sampling_probability > 0:
                weight = 1.0 / sample.sampling_probability
            else:
                weight = 0.0
        else:
            weight = 1.0

        sig_weights[sig] += weight
        total_weight += weight

    # Normalize to get concentrations
    if total_weight > 0:
        return {sig: w / total_weight for sig, w in sig_weights.items()}
    else:
        return {sig: 0.0 for sig in sig_weights}


# =========================
# Parallel ESA Sampling
# =========================


def _esa_worker(
    G_data: Tuple,
    k: int,
    max_retries: int,
    n_samples: int,
    seed_offset: int,
) -> List[Tuple[int, ...]]:
    """Worker function for parallel ESA sampling.

    Args:
        G_data: Tuple of (nodes, edges, is_directed) to reconstruct graph
        k: Subgraph size
        max_retries: Max retries per sample
        n_samples: Number of samples this worker should produce
        seed_offset: Random seed for this worker

    Returns:
        List of vertex tuples (sampled subgraphs)
    """
    import random
    import networkx as nx

    # Reconstruct graph
    nodes, edges, is_directed = G_data
    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Set seed for this worker
    random.seed(seed_offset)

    # Cache edge list
    edge_list = list(G.edges())
    if not edge_list:
        return []

    results = []
    produced = 0
    attempts = 0
    max_attempts = n_samples * (max_retries + 1) * 2  # Safety limit

    while produced < n_samples and attempts < max_attempts:
        attempts += 1

        # Pick random edge
        initial_edge = random.choice(edge_list)
        u, v = initial_edge
        if u == v:  # Skip self-loops
            continue

        sub = {u, v}

        # Expand to k vertices
        stalled = False
        while len(sub) < k:
            frontier_edges = []
            for x in sub:
                # Get neighbors (weak connectivity for directed)
                if is_directed:
                    neighbors = set(G.predecessors(x)) | set(G.successors(x))
                else:
                    neighbors = set(G.neighbors(x))
                for y in neighbors:
                    if y not in sub:
                        frontier_edges.append((x, y))

            if not frontier_edges:
                stalled = True
                break

            chosen_edge = random.choice(frontier_edges)
            _, new_vertex = chosen_edge
            sub.add(new_vertex)

        if not stalled and len(sub) == k:
            results.append(tuple(sub))
            produced += 1

    return results


def parallel_esa_sample(
    G: nx.Graph,
    params: ESAParams,
    n_samples: int,
    num_cores: int | None = None,
) -> List[Tuple[int, ...]]:
    """Parallel ESA sampling across multiple cores.

    Distributes sample collection across workers for speedup on large sample counts.

    Args:
        G: NetworkX graph to sample from
        params: ESAParams with k and max_retries
        n_samples: Total number of samples to collect
        num_cores: Number of CPU cores (default: cpu_count - 2)

    Returns:
        List of vertex tuples (sampled subgraphs)
    """
    import os
    from multiprocessing import Pool
    import random

    if num_cores is None:
        num_cores = max(1, (os.cpu_count() or 4) - 2)

    # For small sample counts, don't bother with parallelization overhead
    if n_samples < num_cores * 10:
        return list(sample_many(G, params, n_samples))

    # Prepare serializable graph data
    G_data = (list(G.nodes()), list(G.edges()), G.is_directed())

    # Distribute samples across workers
    samples_per_worker = n_samples // num_cores
    remainder = n_samples % num_cores

    # Create worker arguments
    base_seed = random.randint(0, 2**31)
    worker_args = []
    for i in range(num_cores):
        worker_samples = samples_per_worker + (1 if i < remainder else 0)
        if worker_samples > 0:
            worker_args.append(
                (
                    G_data,
                    params.k,
                    params.max_retries,
                    worker_samples,
                    base_seed + i,
                )
            )

    # Run parallel sampling
    all_samples = []
    with Pool(processes=num_cores) as pool:
        for worker_results in pool.starmap(_esa_worker, worker_args):
            all_samples.extend(worker_results)

    return all_samples
