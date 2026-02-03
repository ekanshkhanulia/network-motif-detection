from __future__ import annotations

from typing import Optional, List, Tuple
import random
import networkx as nx


def _random_switch_count(edge_count: int, switch_factor: int, rng: random.Random) -> int:
    if edge_count <= 1:
        return 0
    switches_range = switch_factor * edge_count
    if switches_range <= 0:
        return 0
    return int(switches_range + rng.randrange(int(switches_range)))


def _split_directed_edges(
    edge_set: set[tuple[int, int]],
) -> tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    mutual_pairs: List[Tuple[int, int]] = []
    single_edges: List[Tuple[int, int]] = []
    for u, v in edge_set:
        if u == v:
            continue
        if (v, u) in edge_set:
            if u < v:
                mutual_pairs.append((u, v))
        else:
            single_edges.append((u, v))
    return mutual_pairs, single_edges


def _swap_mutual_edges(
    H: nx.DiGraph,
    mutual_pairs: List[Tuple[int, int]],
    edge_set: set[tuple[int, int]],
    num_switches: int,
    rng: random.Random,
) -> None:
    for _ in range(num_switches):
        if len(mutual_pairs) < 2:
            return
        idx1 = rng.randrange(len(mutual_pairs))
        idx2 = rng.randrange(len(mutual_pairs) - 1)
        if idx2 >= idx1:
            idx2 += 1
        a, b = mutual_pairs[idx1]
        c, d = mutual_pairs[idx2]
        if len({a, b, c, d}) < 4:
            continue
        s1, t1 = (a, b) if rng.random() < 0.5 else (b, a)
        s2, t2 = (c, d) if rng.random() < 0.5 else (d, c)
        if (
            (s1, t2) in edge_set
            or (t2, s1) in edge_set
            or (s2, t1) in edge_set
            or (t1, s2) in edge_set
        ):
            continue
        for u, v in ((a, b), (b, a), (c, d), (d, c)):
            edge_set.remove((u, v))
            H.remove_edge(u, v)
        new1 = (s1, t2)
        new2 = (s2, t1)
        for u, v in (new1, new2):
            H.add_edge(u, v)
            H.add_edge(v, u)
            edge_set.add((u, v))
            edge_set.add((v, u))
        mutual_pairs[idx1] = tuple(sorted(new1))
        mutual_pairs[idx2] = tuple(sorted(new2))


def _swap_single_edges(
    H: nx.DiGraph,
    single_edges: List[Tuple[int, int]],
    edge_set: set[tuple[int, int]],
    num_switches: int,
    rng: random.Random,
) -> None:
    for _ in range(num_switches):
        if len(single_edges) < 2:
            return
        idx1 = rng.randrange(len(single_edges))
        idx2 = rng.randrange(len(single_edges) - 1)
        if idx2 >= idx1:
            idx2 += 1
        s1, t1 = single_edges[idx1]
        s2, t2 = single_edges[idx2]
        if (s1 == s2) or (s1 == t2) or (t1 == s2) or (t1 == t2):
            continue
        if (
            (s1, t2) in edge_set
            or (s2, t1) in edge_set
            or (t2, s1) in edge_set
            or (t1, s2) in edge_set
        ):
            continue
        edge_set.remove((s1, t1))
        edge_set.remove((s2, t2))
        H.remove_edge(s1, t1)
        H.remove_edge(s2, t2)
        H.add_edge(s1, t2)
        H.add_edge(s2, t1)
        edge_set.add((s1, t2))
        edge_set.add((s2, t1))
        single_edges[idx1] = (s1, t2)
        single_edges[idx2] = (s2, t1)


def _randomize_directed_mfinder(
    H: nx.DiGraph,
    switch_factor: int,
    rng: random.Random,
) -> None:
    edge_set = set(H.edges())
    mutual_pairs, single_edges = _split_directed_edges(edge_set)
    double_edge_count = 2 * len(mutual_pairs)
    single_edge_count = len(single_edges)
    num_mutual_switches = _random_switch_count(double_edge_count, switch_factor, rng)
    num_single_switches = _random_switch_count(single_edge_count, switch_factor, rng)
    if num_mutual_switches:
        _swap_mutual_edges(H, mutual_pairs, edge_set, num_mutual_switches, rng)
    if num_single_switches:
        _swap_single_edges(H, single_edges, edge_set, num_single_switches, rng)


def _undirected_switch_count(edge_count: int, switch_factor: int, rng: random.Random) -> int:
    if edge_count <= 1:
        return 0
    return (switch_factor + rng.randrange(switch_factor)) * edge_count


def _randomize_undirected_mfinder(
    H: nx.Graph,
    switch_factor: int,
    rng: random.Random,
) -> None:
    edge_set: set[tuple[int, int]] = {tuple(sorted(edge)) for edge in H.edges()}
    edges: List[Tuple[int, int]] = list(edge_set)
    num_switches = _undirected_switch_count(len(edges), switch_factor, rng)
    for _ in range(num_switches):
        if len(edges) < 2:
            return
        idx1 = rng.randrange(len(edges))
        idx2 = rng.randrange(len(edges) - 1)
        if idx2 >= idx1:
            idx2 += 1
        s1, t1 = edges[idx1]
        s2, t2 = edges[idx2]
        if len({s1, t1, s2, t2}) < 4:
            continue
        if rng.random() <= 0.5:
            if (s1 == t2) or (s2 == t1):
                continue
            new1 = tuple(sorted((s1, t2)))
            new2 = tuple(sorted((s2, t1)))
        else:
            if (s1 == s2) or (t1 == t2):
                continue
            new1 = tuple(sorted((s1, s2)))
            new2 = tuple(sorted((t1, t2)))
        if new1 in edge_set or new2 in edge_set:
            continue
        edge_set.remove(edges[idx1])
        edge_set.remove(edges[idx2])
        H.remove_edge(*edges[idx1])
        H.remove_edge(*edges[idx2])
        H.add_edge(*new1)
        H.add_edge(*new2)
        edge_set.add(new1)
        edge_set.add(new2)
        edges[idx1] = new1
        edges[idx2] = new2


def randomize_graph_degree_preserving(
    G: nx.Graph,
    swaps_per_edge: Optional[int] = None,
    tries_per_swap: int = 3,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Return a degree-sequence preserving randomized copy via edge swaps.

    Mimics mfinder's switch-based randomization:
    - Directed graphs conserve mutual edges by default.
    - Switch counts follow the randomized ranges in mfinder.

    Args:
        G: input graph (Graph or DiGraph)
        swaps_per_edge: switch factor (mfinder defaults: 100 directed, 10 undirected)
        tries_per_swap: retained for API compatibility (unused)
        seed: random seed
    """
    _ = tries_per_swap
    rng = random.Random(seed)
    H = G.copy()
    if H.is_directed():
        switch_factor = swaps_per_edge if swaps_per_edge is not None else 100
        _randomize_directed_mfinder(H, switch_factor, rng)
    else:
        switch_factor = swaps_per_edge if swaps_per_edge is not None else 10
        _randomize_undirected_mfinder(H, switch_factor, rng)
    return H
