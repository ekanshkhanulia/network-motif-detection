from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Literal, Tuple, Dict
import math
from multiprocessing import Pool
from functools import partial

import networkx as nx


@dataclass
class RandESUParams:
    k: int  # motif size
    p_depth: Sequence[float]  # probabilities per depth index starting at 1
    child_selection: Literal["bernoulli", "balanced"] = "bernoulli"
    max_degree: int | None = None  # Optional: skip nodes with degree > max_degree

    def __post_init__(self):
        if len(self.p_depth) != self.k:
            raise ValueError(
                f"Need k probabilities (depth 1..k); got {len(self.p_depth)} for k={self.k}"
            )
        if not all(0 < p <= 1 for p in self.p_depth):
            raise ValueError("All p_depth entries must be in (0,1]")
        if self.max_degree is not None and self.max_degree <= 0:
            raise ValueError(f"max_degree must be positive, got {self.max_degree}")


def esu_enumerate(G: nx.Graph, k: int) -> Iterator[Sequence[int]]:
    """Deterministically enumerate all connected induced subgraphs of size k using ESU logic.

    Implementation follows Wernicke (2005) Algorithm ESU exactly:

    For each vertex v:
        V_Extension <- {u in N(v) | u > v}
        ExtendSubgraph({v}, V_Extension, v)

    ExtendSubgraph(V_Subgraph, V_Extension, v):
        if |V_Subgraph| = k: output and return
        while V_Extension not empty:
            Remove w from V_Extension
            V'_Extension <- V_Extension ∪ {u in N_excl(w, V_Subgraph) | u > v}
            ExtendSubgraph(V_Subgraph ∪ {w}, V'_Extension, v)

    Where N_excl(w, V_Subgraph) = N(w) - N(V_Subgraph) are neighbors of w
    that are NOT neighbors of any vertex already in V_Subgraph.
    """

    # Precompute adjacency for speed (use weak connectivity for directed graphs)
    if G.is_directed():
        neighbors = {
            v: set(G.predecessors(v)) | set(G.successors(v)) for v in G.nodes()
        }
    else:
        neighbors = {v: set(G.neighbors(v)) for v in G.nodes()}

    def extend(sub: List[int], ext_cand: set[int], v: int):
        """
        Args:
            sub: Current V_Subgraph (list of vertices)
            ext_cand: Current V_Extension set
            v: The root vertex (for label comparison u > v)
        """
        if len(sub) == k:
            yield tuple(sub)
            return

        # Work through extension candidates
        ext_cand_remaining = ext_cand.copy()
        while ext_cand_remaining:
            # Remove arbitrary vertex w from extension set
            w = ext_cand_remaining.pop()

            # Compute N_excl(w, V_Subgraph): neighbors of w NOT adjacent to any vertex in sub
            # Note: sub already contains the current subgraph vertices
            sub_neighbors = set()
            for node in sub:
                sub_neighbors |= neighbors[node]

            # N_excl(w, sub) = N(w) - N(sub)
            # But we need neighbors of w that are NOT neighbors of V_Subgraph
            n_excl_w = neighbors[w] - sub_neighbors

            # Build new extension: current remaining ∪ {u in N_excl(w, sub) | u > v}
            new_ext = ext_cand_remaining.copy()
            for u in n_excl_w:
                if u > v and u not in sub and u != w:
                    new_ext.add(u)

            # Recurse with V_Subgraph ∪ {w}
            yield from extend(sub + [w], new_ext, v)

    for v in G.nodes():
        # Initial extension: neighbors of v with label > v
        ext_candidates: set[int] = {u for u in neighbors[v] if u > v}
        yield from extend([v], ext_candidates, v)


def rand_esu_sample(G: nx.Graph, params: RandESUParams) -> Iterator[Sequence[int]]:
    """Randomized ESU (RAND-ESU) sampling of connected size-k induced subgraphs.

    Implements RAND-ESU from Wernicke (2005):
    - Each recursive expansion is taken with probability p_depth[d-1] where d = current depth
    - All leaves are sampled with probability prod(p_depth) (Lemma 3)
    - The naive frequency estimator is unbiased (Theorem 4)

    Supports two child selection modes:
    - "bernoulli": Independent coin flip for each child (simpler, higher variance)
    - "balanced": Select exactly floor/ceil(x*p) children (lower variance, per footnote 4)
    """
    k = params.k
    p_depth = params.p_depth
    if G.is_directed():
        neighbors = {
            v: set(G.predecessors(v)) | set(G.successors(v)) for v in G.nodes()
        }
    else:
        neighbors = {v: set(G.neighbors(v)) for v in G.nodes()}

    def extend(sub: List[int], ext_cand: set[int], v: int):
        """
        Args:
            sub: Current V_Subgraph (list of vertices)
            ext_cand: Current V_Extension set
            v: The root vertex (for label comparison u > v)
        """
        current_size = len(sub)
        if current_size == k:
            yield tuple(sub)
            return

        depth_index = (
            current_size  # depth after adding = current_size + 1, but 0-indexed
        )
        p = p_depth[depth_index]

        # Convert to list for sampling
        ext_cand_list = list(ext_cand)

        if params.child_selection == "bernoulli":
            # Independent coin flip for each candidate
            chosen = [w for w in ext_cand_list if random.random() <= p]
        else:
            # Balanced: choose floor/ceil(x*p) children (footnote 4 in paper)
            x = len(ext_cand_list)
            xp = x * p
            low = math.floor(xp)
            high = math.ceil(xp)
            prob_high = xp - low
            k_choose = high if random.random() < prob_high else low
            chosen = random.sample(ext_cand_list, k=min(k_choose, x)) if x > 0 else []

        # Process chosen children in order (simulating the while loop)
        ext_cand_remaining = ext_cand.copy()
        for w in chosen:
            ext_cand_remaining.discard(w)

            # Compute N_excl(w, V_Subgraph): neighbors of w NOT adjacent to any vertex in sub
            sub_neighbors = set()
            for node in sub:
                sub_neighbors |= neighbors[node]
            n_excl_w = neighbors[w] - sub_neighbors

            # Build new extension: remaining ∪ {u in N_excl(w, sub) | u > v}
            new_ext = ext_cand_remaining.copy()
            for u in n_excl_w:
                if u > v and u not in sub and u != w:
                    new_ext.add(u)

            yield from extend(sub + [w], new_ext, v)

    node_list = list(G.nodes())

    # Filter out high-degree nodes if max_degree is specified
    if params.max_degree is not None:
        degrees = dict(G.degree())
        filtered_count = sum(1 for v in node_list if degrees[v] > params.max_degree)
        node_list = [v for v in node_list if degrees[v] <= params.max_degree]
        if filtered_count > 0:
            print(
                f"    [degree filter] Excluded {filtered_count} nodes with degree > {params.max_degree} ({filtered_count / G.number_of_nodes() * 100:.2f}%)",
                flush=True,
            )

    # Select root vertices with probability p_depth[0]
    if params.child_selection == "bernoulli":
        chosen_roots = [v for v in node_list if random.random() <= p_depth[0]]
    else:
        x = len(node_list)
        xp = x * p_depth[0]
        low = math.floor(xp)
        high = math.ceil(xp)
        prob_high = xp - low
        k_choose = high if random.random() < prob_high else low
        chosen_roots = random.sample(node_list, k=min(k_choose, x)) if x > 0 else []

    for v in chosen_roots:
        # Initial extension: neighbors of v with label > v
        ext_candidates: set[int] = {u for u in neighbors[v] if u > v}
        yield from extend([v], ext_candidates, v)


def approximate_realized_fraction(
    k: int,
    p_depth: Sequence[float],
    counted_leaves: int,
    total_possible_leaves: int | None,
) -> Tuple[float, Dict[str, float]]:
    """Compute realized sampling fraction diagnostics.

    If total_possible_leaves provided (from enumeration), realized = counted_leaves / total_possible_leaves.
    Otherwise return theoretical target product and leaf count only.

    Returns (realized_fraction_est, details_dict).
    """
    target = 1.0
    for p in p_depth:
        target *= p
    if total_possible_leaves and total_possible_leaves > 0:
        realized = counted_leaves / total_possible_leaves
    else:
        realized = float("nan")
    details = {
        "theoretical_target_fraction": target,
        "counted_leaves": counted_leaves,
        "total_possible_leaves": total_possible_leaves
        if total_possible_leaves is not None
        else -1,
        "realized_fraction": realized,
        "fraction_error": (realized - target)
        if (not (total_possible_leaves is None) and not math.isnan(realized))
        else float("nan"),
    }
    return realized, details


def _sample_from_roots(roots_chunk, G_data, params, seed_offset):
    """
    Worker function for parallel sampling. Samples from a subset of root nodes.

    Args:
        roots_chunk: List of root nodes to sample from
        G_data: Tuple of (nodes, edges, is_directed) to reconstruct graph
        params: RandESUParams
        seed_offset: Random seed offset for this worker

    Returns:
        List of sampled subgraphs
    """
    # Reconstruct graph from data
    nodes, edges, is_directed = G_data
    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Set random seed for reproducibility (different for each worker)
    random.seed(seed_offset)

    # Build neighbor dict
    if G.is_directed():
        neighbors = {
            v: set(G.predecessors(v)) | set(G.successors(v)) for v in G.nodes()
        }
    else:
        neighbors = {v: set(G.neighbors(v)) for v in G.nodes()}

    k = params.k
    p_depth = params.p_depth

    def extend(sub: List[int], ext_cand: set[int], v: int):
        """
        Args:
            sub: Current V_Subgraph
            ext_cand: Current V_Extension set
            v: Root vertex for label comparison
        """
        current_size = len(sub)
        if current_size == k:
            yield tuple(sub)
            return

        depth_index = current_size
        p = p_depth[depth_index]
        ext_cand_list = list(ext_cand)

        if params.child_selection == "bernoulli":
            chosen = [w for w in ext_cand_list if random.random() <= p]
        else:
            import math

            x = len(ext_cand_list)
            xp = x * p
            low = math.floor(xp)
            high = math.ceil(xp)
            prob_high = xp - low
            k_choose = high if random.random() < prob_high else low
            chosen = random.sample(ext_cand_list, k=min(k_choose, x)) if x > 0 else []

        # Process chosen children
        ext_cand_remaining = ext_cand.copy()
        for w in chosen:
            ext_cand_remaining.discard(w)

            # Compute N_excl(w, V_Subgraph)
            sub_neighbors = set()
            for node in sub:
                sub_neighbors |= neighbors[node]
            n_excl_w = neighbors[w] - sub_neighbors

            # Build new extension
            new_ext = ext_cand_remaining.copy()
            for u in n_excl_w:
                if u > v and u not in sub and u != w:
                    new_ext.add(u)

            yield from extend(sub + [w], new_ext, v)

    # Sample from assigned roots
    results = []
    for v in roots_chunk:
        ext_candidates: set[int] = {u for u in neighbors[v] if u > v}
        for subgraph in extend([v], ext_candidates, v):
            results.append(subgraph)

    return results


def parallel_rand_esu_sample(
    G: nx.Graph,
    params: RandESUParams,
    num_cores: int | None = None,
    progress_callback=None,
) -> Iterator[Sequence[int]]:
    """
    Parallel version of rand_esu_sample using multiprocessing with chunked work distribution.

    Divides root nodes into small chunks for better load balancing across workers.

    Args:
        G: NetworkX graph
        params: RandESUParams with k, p_depth, child_selection
        num_cores: Number of CPU cores to use (default: cpu_count - 2)
        progress_callback: Optional function(completed, total, elapsed_sec, chunk_time_sec, chunk_info=None)
                          called after each chunk with timing information. chunk_info dict passed
                          at start with configuration details.

    Yields:
        Sampled subgraphs (same format as rand_esu_sample)
    """
    import os

    if num_cores is None:
        num_cores = max(1, (os.cpu_count() or 4) - 2)

    k = params.k
    p_depth = params.p_depth

    # Select root nodes (same logic as rand_esu_sample)
    node_list = list(G.nodes())

    # Filter out high-degree nodes if max_degree is specified
    if params.max_degree is not None:
        degrees = dict(G.degree())
        filtered_count = sum(1 for v in node_list if degrees[v] > params.max_degree)
        node_list = [v for v in node_list if degrees[v] <= params.max_degree]
        if filtered_count > 0:
            print(
                f"    [degree filter] Excluded {filtered_count} nodes with degree > {params.max_degree} ({filtered_count / G.number_of_nodes() * 100:.2f}%)",
                flush=True,
            )

    if params.child_selection == "bernoulli":
        chosen_roots = [v for v in node_list if random.random() <= p_depth[0]]
    else:
        import math

        x = len(node_list)
        xp = x * p_depth[0]
        low = math.floor(xp)
        high = math.ceil(xp)
        prob_high = xp - low
        k_choose = high if random.random() < prob_high else low
        chosen_roots = random.sample(node_list, k=min(k_choose, x)) if x > 0 else []

    # If very few roots, don't bother with parallelization
    if len(chosen_roots) < num_cores * 2:
        # Fall back to sequential
        yield from rand_esu_sample(G, params)
        return

    # Prepare graph data for serialization (graphs can't be pickled easily)
    G_data = (list(G.nodes()), list(G.edges()), G.is_directed())

    # Create smaller chunks for better load balancing
    # Use 4x more chunks than cores so workers can pick up new work as they finish
    # Reduced from 8x to 4x to minimize process spawning overhead
    num_chunks = min(num_cores * 4, len(chosen_roots))
    chunk_size = max(1, (len(chosen_roots) + num_chunks - 1) // num_chunks)

    root_chunks = [
        chosen_roots[i : i + chunk_size]
        for i in range(0, len(chosen_roots), chunk_size)
    ]

    # Notify about chunk configuration if callback provided
    if progress_callback:
        progress_callback(
            0,
            len(root_chunks),
            0,
            0,
            chunk_info={
                "num_chunks": len(root_chunks),
                "avg_chunk_size": chunk_size,
                "total_roots": len(chosen_roots),
            },
        )

    # Create different seeds for each chunk for reproducibility
    base_seed = random.randint(0, 2**31)

    # Run parallel sampling with imap_unordered for progress tracking
    with Pool(processes=num_cores) as pool:
        # Map each chunk to a worker with different seed
        worker_args = [
            (chunk, G_data, params, base_seed + i)
            for i, chunk in enumerate(root_chunks)
        ]

        # Execute in parallel with progress tracking
        import time as time_module

        completed_chunks = 0
        total_chunks = len(root_chunks)
        start_time = time_module.time()
        chunk_start_time = start_time

        for worker_results in pool.starmap(_sample_from_roots, worker_args):
            completed_chunks += 1
            current_time = time_module.time()

            # Call progress callback if provided with timing info
            if progress_callback:
                elapsed = current_time - start_time
                chunk_time = current_time - chunk_start_time
                progress_callback(completed_chunks, total_chunks, elapsed, chunk_time)
                chunk_start_time = current_time

            # Yield results from this chunk
            for subgraph in worker_results:
                yield subgraph


__all__ = [
    "RandESUParams",
    "esu_enumerate",
    "rand_esu_sample",
    "parallel_rand_esu_sample",
    "parallel_esu_count",
    "parallel_rand_esu_count",
    "parallel_esu_enumerate_with_signatures",
    "approximate_realized_fraction",
]


# =========================
# Memory-lean parallel counting
# =========================


def _adjacency_string_from_nodes(G: nx.Graph, node_order: Sequence[int]) -> str:
    """Return adjacency bits for G restricted to node_order without building a subgraph.

    This avoids per-sample Graph allocations, reducing memory pressure.
    """
    k = len(node_order)
    bits: List[str] = []
    if G.is_directed():
        for i in range(k):
            u = node_order[i]
            for j in range(k):
                v = node_order[j]
                bits.append("1" if G.has_edge(u, v) else "0")
    else:
        for i in range(k):
            u = node_order[i]
            for j in range(k):
                v = node_order[j]
                bits.append("1" if G.has_edge(u, v) else "0")
    return "".join(bits)


def _canonical_signature_nodes(G: nx.Graph, nodes: Sequence[int]) -> str:
    """Canonical unlabeled signature by brute-forcing permutations of nodes on G.

    Equivalent to utils.motifs.canonical_signature(G.subgraph(nodes)), but avoids subgraph copies.
    """
    from itertools import permutations as _permutations

    best: str | None = None
    for order in _permutations(nodes):
        s = _adjacency_string_from_nodes(G, order)
        if best is None or s < best:
            best = s
    assert best is not None
    return best


def _count_from_roots(
    roots_chunk: Sequence[int],
    G_data: Tuple[List[int], List[Tuple[int, int]], bool],
    params: RandESUParams,
    seed_offset: int,
) -> Tuple[int, Dict[str, int]]:
    """Worker: count signatures from a chunk of roots without storing subgraphs.

    Returns (total_count, freq_map) to minimize IPC memory.
    """
    # Reconstruct graph
    nodes, edges, is_directed = G_data
    G = nx.DiGraph() if is_directed else nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    random.seed(seed_offset)

    # Precompute neighbors for ESU extension (weak connectivity for DiGraph)
    if G.is_directed():
        neighbors = {
            v: set(G.predecessors(v)) | set(G.successors(v)) for v in G.nodes()
        }
    else:
        neighbors = {v: set(G.neighbors(v)) for v in G.nodes()}

    k = params.k
    p_depth = params.p_depth

    total = 0
    freq: Dict[str, int] = {}

    def extend(sub: List[int], ext_cand: set[int], v: int):
        """
        Args:
            sub: Current V_Subgraph
            ext_cand: Current V_Extension set
            v: Root vertex for label comparison
        """
        nonlocal total, freq
        current_size = len(sub)
        if current_size == k:
            # Count directly
            sig = _canonical_signature_nodes(G, sub)
            freq[sig] = freq.get(sig, 0) + 1
            total += 1
            return

        depth_index = current_size
        p = p_depth[depth_index]
        ext_cand_list = list(ext_cand)

        if params.child_selection == "bernoulli":
            chosen = [w for w in ext_cand_list if random.random() <= p]
        else:
            import math as _math

            x = len(ext_cand_list)
            xp = x * p
            low = _math.floor(xp)
            high = _math.ceil(xp)
            prob_high = xp - low
            k_choose = high if random.random() < prob_high else low
            chosen = random.sample(ext_cand_list, k=min(k_choose, x)) if x > 0 else []

        # Process chosen children
        ext_cand_remaining = ext_cand.copy()
        for w in chosen:
            ext_cand_remaining.discard(w)

            # Compute N_excl(w, V_Subgraph)
            sub_neighbors = set()
            for node in sub:
                sub_neighbors |= neighbors[node]
            n_excl_w = neighbors[w] - sub_neighbors

            # Build new extension
            new_ext = ext_cand_remaining.copy()
            for u in n_excl_w:
                if u > v and u not in sub and u != w:
                    new_ext.add(u)

            extend(sub + [w], new_ext, v)

    # Process assigned roots
    for v in roots_chunk:
        ext_candidates: set[int] = {u for u in neighbors[v] if u > v}
        extend([v], ext_candidates, v)

    return total, freq


def parallel_rand_esu_count(
    G: nx.Graph,
    params: RandESUParams,
    num_cores: int | None = None,
    progress_callback=None,
) -> Tuple[int, Dict[str, int]]:
    """Parallel RAND-ESU that returns counts directly, avoiding large result lists.

    Designed to be memory-friendly at higher k by aggregating counts per worker.

    Args:
        G: NetworkX graph
        params: RandESUParams with k, p_depth, child_selection
        num_cores: Number of CPU cores to use (default: cpu_count - 2)
        progress_callback: Optional callback function

    Returns:
        Tuple of (total_count, freq_dict)
    """
    import os

    if num_cores is None:
        num_cores = max(1, (os.cpu_count() or 4) - 2)

    k = params.k
    p_depth = params.p_depth

    # Select root nodes
    node_list = list(G.nodes())

    # Optional degree filter
    if params.max_degree is not None:
        degrees = dict(G.degree())
        node_list = [v for v in node_list if degrees[v] <= params.max_degree]

    if params.child_selection == "bernoulli":
        chosen_roots = [v for v in node_list if random.random() <= p_depth[0]]
    else:
        import math as _math

        x = len(node_list)
        xp = x * p_depth[0]
        low = _math.floor(xp)
        high = _math.ceil(xp)
        prob_high = xp - low
        k_choose = high if random.random() < prob_high else low
        chosen_roots = random.sample(node_list, k=min(k_choose, x)) if x > 0 else []

    # If little work, do sequential counting locally
    if len(chosen_roots) < num_cores * 2:
        total = 0
        freq: Dict[str, int] = {}

        # Precompute neighbors
        if G.is_directed():
            neighbors = {
                v: set(G.predecessors(v)) | set(G.successors(v)) for v in G.nodes()
            }
        else:
            neighbors = {v: set(G.neighbors(v)) for v in G.nodes()}

        def extend(sub: List[int], ext_cand: set[int], v: int):
            """Sequential fallback extend function."""
            nonlocal total, freq
            current_size = len(sub)
            if current_size == k:
                sig = _canonical_signature_nodes(G, sub)
                freq[sig] = freq.get(sig, 0) + 1
                total += 1
                return

            depth_index = current_size
            p = p_depth[depth_index]
            ext_cand_list = list(ext_cand)

            if params.child_selection == "bernoulli":
                chosen = [w for w in ext_cand_list if random.random() <= p]
            else:
                import math as _math

                x = len(ext_cand_list)
                xp = x * p
                low = _math.floor(xp)
                high = _math.ceil(xp)
                prob_high = xp - low
                k_choose = high if random.random() < prob_high else low
                chosen = (
                    random.sample(ext_cand_list, k=min(k_choose, x)) if x > 0 else []
                )

            # Process chosen children
            ext_cand_remaining = ext_cand.copy()
            for w in chosen:
                ext_cand_remaining.discard(w)

                # Compute N_excl(w, V_Subgraph)
                sub_neighbors = set()
                for node in sub:
                    sub_neighbors |= neighbors[node]
                n_excl_w = neighbors[w] - sub_neighbors

                # Build new extension
                new_ext = ext_cand_remaining.copy()
                for u in n_excl_w:
                    if u > v and u not in sub and u != w:
                        new_ext.add(u)

                extend(sub + [w], new_ext, v)

        for v in chosen_roots:
            ext_candidates: set[int] = {u for u in neighbors[v] if u > v}
            extend([v], ext_candidates, v)

        return total, freq

    # Parallel processing: enough roots for parallelization
    # Prepare graph data for serialization
    G_data = (list(G.nodes()), list(G.edges()), G.is_directed())

    # Create smaller chunks for better load balancing
    num_chunks = min(num_cores * 4, len(chosen_roots))
    chunk_size = max(1, (len(chosen_roots) + num_chunks - 1) // num_chunks)

    root_chunks = [
        chosen_roots[i : i + chunk_size]
        for i in range(0, len(chosen_roots), chunk_size)
    ]

    if progress_callback:
        progress_callback(
            0,
            len(root_chunks),
            0,
            0,
            chunk_info={
                "num_chunks": len(root_chunks),
                "avg_chunk_size": chunk_size,
                "total_roots": len(chosen_roots),
            },
        )

    base_seed = random.randint(0, 2**31)

    total = 0
    freq: Dict[str, int] = {}

    import time as _time

    with Pool(processes=num_cores) as pool:
        worker_args = [
            (chunk, G_data, params, base_seed + i)
            for i, chunk in enumerate(root_chunks)
        ]

        completed_chunks = 0
        start_time = _time.time()
        last_chunk_time = start_time

        for worker_total, worker_freq in pool.starmap(_count_from_roots, worker_args):
            # Merge results
            total += worker_total
            for sig, c in worker_freq.items():
                freq[sig] = freq.get(sig, 0) + c

            completed_chunks += 1
            now = _time.time()
            if progress_callback:
                progress_callback(
                    completed_chunks,
                    len(root_chunks),
                    now - start_time,
                    now - last_chunk_time,
                )
                last_chunk_time = now

    return total, freq


# =========================
# Parallel full enumeration (counting only)
# =========================


def _enumerate_from_roots(
    roots_chunk: Sequence[int],
    G_data: Tuple[List[int], List[Tuple[int, int]], bool],
    k: int,
) -> int:
    """Worker: count all k-subgraphs rooted at given vertices.

    Returns total count from this chunk.
    """
    # Reconstruct graph
    nodes, edges, is_directed = G_data
    if is_directed:
        import networkx as nx

        G = nx.DiGraph()
    else:
        import networkx as nx

        G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Precompute neighbors for ESU extension (weak connectivity for DiGraph)
    if is_directed:
        neighbors = {
            v: set(G.predecessors(v)) | set(G.successors(v)) for v in G.nodes()
        }
    else:
        neighbors = {v: set(G.neighbors(v)) for v in G.nodes()}

    count = 0

    def extend(sub: List[int], ext_cand: set, v: int):
        nonlocal count
        if len(sub) == k:
            count += 1
            return

        ext_cand_remaining = ext_cand.copy()
        while ext_cand_remaining:
            w = ext_cand_remaining.pop()

            # Compute N_excl(w, V_Subgraph)
            sub_neighbors = set()
            for node in sub:
                sub_neighbors |= neighbors[node]
            n_excl_w = neighbors[w] - sub_neighbors

            # Build new extension
            new_ext = ext_cand_remaining.copy()
            for u in n_excl_w:
                if u > v and u not in sub and u != w:
                    new_ext.add(u)

            extend(sub + [w], new_ext, v)

    # Process assigned roots
    for v in roots_chunk:
        ext_candidates: set = {u for u in neighbors[v] if u > v}
        extend([v], ext_candidates, v)

    return count


def parallel_esu_count(
    G,
    k: int,
    num_cores: int | None = None,
    progress_callback=None,
) -> int:
    """Parallel full enumeration of all k-subgraphs, returning only the count.

    This is much faster than sequential enumeration for large graphs.

    Args:
        G: NetworkX graph
        k: Subgraph size
        num_cores: Number of CPU cores (default: cpu_count - 2)
        progress_callback: Optional function(completed, total, elapsed_sec, chunk_time_sec)

    Returns:
        Total count of k-subgraphs in G
    """
    import os
    from multiprocessing import Pool as _Pool
    import time as _time

    if num_cores is None:
        num_cores = max(1, (os.cpu_count() or 4) - 2)

    node_list = list(G.nodes())

    # If very few nodes, do sequential counting
    if len(node_list) < num_cores * 2:
        return sum(1 for _ in esu_enumerate(G, k))

    # Prepare serializable graph data
    G_data = (list(G.nodes()), list(G.edges()), G.is_directed())

    # Chunk roots for load balancing (use more chunks for better distribution)
    num_chunks = min(num_cores * 4, len(node_list))
    chunk_size = max(1, (len(node_list) + num_chunks - 1) // num_chunks)
    root_chunks = [
        node_list[i : i + chunk_size] for i in range(0, len(node_list), chunk_size)
    ]

    if progress_callback:
        progress_callback(
            0,
            len(root_chunks),
            0,
            0,
            chunk_info={
                "num_chunks": len(root_chunks),
                "avg_chunk_size": chunk_size,
                "total_roots": len(node_list),
            },
        )

    total = 0

    with _Pool(processes=num_cores) as pool:
        worker_args = [(chunk, G_data, k) for chunk in root_chunks]

        completed_chunks = 0
        start_time = _time.time()
        last_chunk_time = start_time

        for worker_count in pool.starmap(_enumerate_from_roots, worker_args):
            total += worker_count

            completed_chunks += 1
            now = _time.time()
            if progress_callback:
                progress_callback(
                    completed_chunks,
                    len(root_chunks),
                    now - start_time,
                    now - last_chunk_time,
                )
                last_chunk_time = now

    return total


def _enumerate_with_sigs_from_roots(
    roots_chunk: Sequence[int],
    G_data: Tuple[List[int], List[Tuple[int, int]], bool],
    k: int,
) -> Tuple[int, Dict[str, int]]:
    """Worker: enumerate all k-subgraphs rooted at given vertices and compute signatures.

    Returns (total_count, freq_dict) from this chunk.
    """
    from itertools import permutations

    # Reconstruct graph
    nodes, edges, is_directed = G_data
    if is_directed:
        import networkx as nx

        G = nx.DiGraph()
    else:
        import networkx as nx

        G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Precompute neighbors for ESU extension (weak connectivity for DiGraph)
    if is_directed:
        neighbors = {
            v: set(G.predecessors(v)) | set(G.successors(v)) for v in G.nodes()
        }
    else:
        neighbors = {v: set(G.neighbors(v)) for v in G.nodes()}

    count = 0
    freq: Dict[str, int] = {}

    def adjacency_string(node_order: Sequence[int]) -> str:
        """Compute adjacency string for induced subgraph without allocations."""
        bits = []
        for i in range(len(node_order)):
            u = node_order[i]
            for j in range(len(node_order)):
                v = node_order[j]
                bits.append("1" if G.has_edge(u, v) else "0")
        return "".join(bits)

    def canonical_signature(node_list: List[int]) -> str:
        """Compute canonical signature by trying all permutations."""
        best = None
        for order in permutations(node_list):
            s = adjacency_string(order)
            if best is None or s < best:
                best = s
        return best

    def extend(sub: List[int], ext_cand: set, v: int):
        nonlocal count, freq
        if len(sub) == k:
            sig = canonical_signature(sub)
            freq[sig] = freq.get(sig, 0) + 1
            count += 1
            return

        ext_cand_remaining = ext_cand.copy()
        while ext_cand_remaining:
            w = ext_cand_remaining.pop()

            # Compute N_excl(w, V_Subgraph)
            sub_neighbors = set()
            for node in sub:
                sub_neighbors |= neighbors[node]
            n_excl_w = neighbors[w] - sub_neighbors

            # Build new extension
            new_ext = ext_cand_remaining.copy()
            for u in n_excl_w:
                if u > v and u not in sub and u != w:
                    new_ext.add(u)

            extend(sub + [w], new_ext, v)

    # Process assigned roots
    for v in roots_chunk:
        ext_candidates: set = {u for u in neighbors[v] if u > v}
        extend([v], ext_candidates, v)

    return count, freq


def parallel_esu_enumerate_with_signatures(
    G,
    k: int,
    num_cores: int | None = None,
    progress_callback=None,
) -> Tuple[int, Dict[str, int]]:
    """Parallel full enumeration of all k-subgraphs, returning count and signature frequencies.

    This is the parallel version suitable for edge-swap significance testing where
    we need both the total count and the frequency of each motif signature.

    Args:
        G: NetworkX graph
        k: Subgraph size
        num_cores: Number of CPU cores (default: cpu_count - 2)
        progress_callback: Optional function(completed, total, elapsed_sec, chunk_time_sec)

    Returns:
        (total_count, freq_dict) where freq_dict maps signatures to counts
    """
    import os
    from multiprocessing import Pool as _Pool
    import time as _time

    if num_cores is None:
        num_cores = max(1, (os.cpu_count() or 4) - 2)

    node_list = list(G.nodes())

    # If very few nodes, do sequential enumeration
    if len(node_list) < num_cores * 2:
        from src.utils.motifs import count_motif_signatures

        return count_motif_signatures(G, esu_enumerate(G, k))

    # Prepare serializable graph data
    G_data = (list(G.nodes()), list(G.edges()), G.is_directed())

    # Chunk roots for load balancing (use more chunks for better distribution)
    num_chunks = min(num_cores * 4, len(node_list))
    chunk_size = max(1, (len(node_list) + num_chunks - 1) // num_chunks)
    root_chunks = [
        node_list[i : i + chunk_size] for i in range(0, len(node_list), chunk_size)
    ]

    if progress_callback:
        progress_callback(
            0,
            len(root_chunks),
            0,
            0,
            chunk_info={
                "num_chunks": len(root_chunks),
                "avg_chunk_size": chunk_size,
                "total_roots": len(node_list),
            },
        )

    total = 0
    freq: Dict[str, int] = {}

    with _Pool(processes=num_cores) as pool:
        worker_args = [(chunk, G_data, k) for chunk in root_chunks]

        completed_chunks = 0
        start_time = _time.time()
        last_chunk_time = start_time

        for worker_total, worker_freq in pool.starmap(
            _enumerate_with_sigs_from_roots, worker_args
        ):
            # Merge results
            total += worker_total
            for sig, c in worker_freq.items():
                freq[sig] = freq.get(sig, 0) + c

            completed_chunks += 1
            now = _time.time()
            if progress_callback:
                progress_callback(
                    completed_chunks,
                    len(root_chunks),
                    now - start_time,
                    now - last_chunk_time,
                )
                last_chunk_time = now

    return total, freq
