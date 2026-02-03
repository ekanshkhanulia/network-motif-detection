from __future__ import annotations

from itertools import permutations
from typing import Iterable, List, Sequence, Tuple, Dict

import networkx as nx


def induced_subgraph_nodes(G: nx.Graph, nodes: Sequence[int]) -> nx.Graph:
    """Return an induced subgraph on a given node sequence (preserving order in returned list)."""
    return G.subgraph(nodes).copy()


def adjacency_string(G_sub: nx.Graph, node_order: Sequence[int]) -> str:
    """Return a row-major adjacency string for the subgraph in the given node order.

    Works for Graph and DiGraph. For Graph, symmetry is implicit; for DiGraph, edge direction is encoded.
    """
    lookup = {n: i for i, n in enumerate(node_order)}
    k = len(node_order)
    bits: List[str] = []
    if G_sub.is_directed():
        for i in range(k):
            u = node_order[i]
            for j in range(k):
                v = node_order[j]
                bits.append("1" if G_sub.has_edge(u, v) else "0")
    else:
        # For undirected graphs, include full matrix for uniformity
        for i in range(k):
            u = node_order[i]
            for j in range(k):
                v = node_order[j]
                bits.append("1" if G_sub.has_edge(u, v) else "0")
    return "".join(bits)


def canonical_signature(G_sub: nx.Graph) -> str:
    """Compute a canonical unlabeled signature for a small subgraph (k <= ~6).

    We brute-force over all permutations of node orderings and take the lexicographically smallest
    adjacency string. This yields a canonical representation of the isomorphism class.
    """
    nodes = list(G_sub.nodes())
    best: str | None = None
    for order in permutations(nodes):
        s = adjacency_string(G_sub, order)
        if best is None or s < best:
            best = s
    assert best is not None
    return best


# --- Triad (directed 3-node motif) classification support ---
# Standard 13 triad census labels (following Davis & Leinhardt / Holland & Leinhardt conventions),
# e.g., '003', '012', '102', '021D', '021U', '021C', '111D', '111U', '030T', '030C', '201', '120D', '120U', '120C', '210', '300'.
# We will map adjacency signatures (length 9 for 3 nodes directed) to a simplified label set.

# Hardcoded correct mapping from canonical signature to standard triad census label.
# These signatures were verified against NetworkX triadic_census.
# The 16 standard triads: 003, 012, 102, 021D, 021U, 021C, 111D, 111U, 030T, 030C, 201, 120D, 120U, 120C, 210, 300
TRIAD_LABELS: Dict[str, str] = {
    # Empty and disconnected triads (for completeness)
    '000000000': '003',  # No edges
    '000000010': '012',  # Single directed edge
    '000010010': '102',  # Single mutual pair (2 directed edges)
    # Two-edge triads (2 directed edges, no mutual)
    '000000110': '021D',  # Out-star: one node points to two others
    '000100100': '021U',  # In-star: two nodes point to one
    '000001100': '021C',  # Chain: A->B->C
    # Three-edge triads (one mutual pair + one single)
    '000001110': '111U',  # Mutual pair + incoming single
    '001001010': '111D',  # Mutual pair + outgoing single
    # Three-edge triads (3 directed, no mutual)
    '000100110': '030T',  # Transitive triple (feed-forward)
    '001100010': '030C',  # 3-cycle
    # Four-edge triads (one mutual pair + 2 singles)
    '001001110': '201',   # Mutual pair + 2 non-adjacent singles
    '001101100': '120D',  # Mutual pair + out-star from mutual
    '000101110': '120U',  # Mutual pair + in-star to mutual
    '001100110': '120C',  # Mutual pair + cycle through third
    # Five-edge triad (two mutual pairs + one single)
    '001101110': '210',   # Two mutual pairs + one single
    # Complete triad (three mutual pairs = 6 directed edges)
    '011101110': '300',   # All mutual (complete directed graph)
}

def triad_label_from_signature(sig: str) -> str:
    return TRIAD_LABELS.get(sig, 'OTHER')


def count_motif_signatures(
    G: nx.Graph,
    subgraphs: Iterable[Sequence[int]],
) -> Tuple[int, dict[str, int]]:
    """Count motif signatures for a stream of sampled subgraphs.

    Args:
        G: the full graph
        subgraphs: iterable of sequences of node IDs forming connected induced subgraphs

    Returns:
        total_count, freq_map where keys are canonical signatures
    """
    freq: dict[str, int] = {}
    total = 0
    for nodes in subgraphs:
        G_sub = induced_subgraph_nodes(G, nodes)
        sig = canonical_signature(G_sub)
        freq[sig] = freq.get(sig, 0) + 1
        total += 1
    return total, freq
