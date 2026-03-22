from __future__ import annotations
from typing import Dict, Iterable, List, Set, Tuple
import math
import numpy as np
from brcore.graph.compact import CompactDiGraph


# Euclidian distance (in meters) between two nodes in int xy coordinates
def _dist_int(xy_int: np.ndarray, a: int, b: int) -> int:
    dx = int(xy_int[a, 0] - xy_int[b, 0])
    dy = int(xy_int[a, 1] - xy_int[b, 1])
    return int(math.isqrt(dx * dx + dy * dy))


def search_space_reduction(
    G: CompactDiGraph,
    xy_int: np.ndarray,
    seeds: List[int],
    partition: Dict[int, int],
    boundary_nodes: Set[int],
    s: int,
    t: int,
    corridor_slack_m: int = 1500,
    max_hops_from_boundary: int = 0,
) -> Tuple[Set[int], Set[int]]:
    """
    Stage-1 reduction:
      1) keep a set of partition cells whose seed lies in a corridor s->t
      2) keep all nodes in those cells
      3) additionally include boundary nodes of those cells + optional hop expansion

    Returns:
      kept_cells, kept_nodes
    """
    if not (0 <= s < G.n_nodes and 0 <= t < G.n_nodes):
        raise ValueError("s or t out of node id range")

    # --- A) corridor on cell seeds ---
    dst = _dist_int(xy_int, s, t)
    kept_cells: Set[int] = set()

    for cid, seed_node in enumerate(seeds):
        d1 = _dist_int(xy_int, s, seed_node)
        d2 = _dist_int(xy_int, seed_node, t)
        if d1 + d2 <= dst + corridor_slack_m:
            kept_cells.add(cid)

    # always keep the cells containing s and t if available
    if s in partition:
        kept_cells.add(partition[s])
    if t in partition:
        kept_cells.add(partition[t])

    # --- B) keep nodes in kept cells ---
    kept_nodes: Set[int] = set()
    for v, cid in partition.items():
        if cid in kept_cells:
            kept_nodes.add(v)

    # --- C) boundary nodes in kept cells ---
    boundary_kept = set()
    for b in boundary_nodes:
        cid = partition.get(b, None)
        if cid is not None and cid in kept_cells:
            boundary_kept.add(b)

    kept_nodes |= boundary_kept

    # --- D) hop expansion around boundary nodes (optional) ---
    if max_hops_from_boundary > 0 and boundary_kept:
        frontier = set(boundary_kept)
        visited = set(boundary_kept)
        for _ in range(max_hops_from_boundary):
            new_frontier = set()
            for u in frontier:
                to, _, _ = G.neighbors(u)
                for v in to:
                    vv = int(v)
                    if vv not in visited:
                        visited.add(vv)
                        new_frontier.add(vv)
            kept_nodes |= new_frontier
            frontier = new_frontier
            if not frontier:
                break

    return kept_cells, kept_nodes
