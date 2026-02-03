from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterable, Optional, Tuple

import networkx as nx


def _iter_edges_from_file(path: Path) -> Iterable[Tuple[str, str]]:
    """Yield edges (u, v) from a SNAP-style edge list.

    SNAP files typically have comment lines starting with '#', and whitespace-separated pairs per line.
    Nodes are arbitrary strings/ids; we keep them as strings then relabel to ints later if needed.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            if u == v:
                # drop self-loops
                continue
            yield u, v


def load_snap_graph(
    path: Path,
    directed: bool,
    max_nodes: Optional[int] = None,
    max_degree: Optional[int] = None,
    relabel_to_integers: bool = True,
) -> nx.Graph:
    """Load a SNAP dataset into a NetworkX Graph/DiGraph.

    Args:
        path: path to edge list (possibly .gz)
        directed: True for DiGraph, False for Graph
        max_nodes: if provided, limit to the first N unique nodes encountered (useful for smoke tests)
        max_degree: if provided, filter out nodes with degree > max_degree after loading
        relabel_to_integers: map node labels to compact consecutive integers starting at 0

    Returns:
        NetworkX Graph or DiGraph
    """
    G = nx.DiGraph() if directed else nx.Graph()

    nodes_seen = set()
    node_limit_reached = False

    for u, v in _iter_edges_from_file(Path(path)):
        if max_nodes is not None:
            if u not in nodes_seen:
                if len(nodes_seen) >= max_nodes:
                    node_limit_reached = True
                else:
                    nodes_seen.add(u)
            if v not in nodes_seen:
                if len(nodes_seen) >= max_nodes:
                    node_limit_reached = True
                else:
                    nodes_seen.add(v)
            if node_limit_reached and (u not in nodes_seen or v not in nodes_seen):
                # Skip edges introducing new nodes beyond cap
                continue
        G.add_edge(u, v)

    # Clean duplicates/self-loops handled implicitly by NX

    if relabel_to_integers:
        mapping = {n: i for i, n in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping, copy=True)

    # Filter high-degree nodes if max_degree specified
    if max_degree is not None and max_degree > 0:
        degrees = dict(G.degree())
        high_degree_nodes = [n for n, d in degrees.items() if d > max_degree]
        if high_degree_nodes:
            G.remove_nodes_from(high_degree_nodes)
            # Note: This may create isolated nodes or disconnect the graph
            # Remove isolated nodes for cleaner graph
            isolated = list(nx.isolates(G))
            if isolated:
                G.remove_nodes_from(isolated)

    return G
