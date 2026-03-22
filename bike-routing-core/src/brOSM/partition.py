"""
seed_partition_pipeline.py

Minimal end-to-end pipeline for:
1) hub-aware candidate selection using edge attribute 'road_id32'
2) k-means++ (D^2) seed selection on candidates
3) optional mini-batch k-means refinement of centers (1 or 2 passes)
4) snap refined centers back to node IDs + deduplicate/fill
5) (optional) hop-bounded BFS partition from these seeds
6) Graph Voronoi partition using hop distance (each edge has cost 1).
7) plotting: seeds, partition, boundaries

Requirements:
- numpy
- scipy (for cKDTree)
- matplotlib
- osmnx
- networkx (pulled by osmnx)
- scikit-learn is NOT required in this file (we implement D^2 sampling ourselves)

Usage (example):
    import osmnx as ox
    from seed_partition_pipeline import build_seeds, plot_seeds, hop_bounded_bfs_partition, plot_partition, plot_boundaries

    G = ox.graph_from_place("Manhattan, New York, USA", network_type="drive")
    G = ox.project_graph(G)

    seeds = build_seeds(G, K=500, H_factor=50, alpha=0.25, refine=True)
    plot_seeds(G, seeds)

    P = hop_bounded_bfs_partition(G, k=6, seeds=seeds)
    plot_partition(G, P)
    plot_boundaries(G, P)
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from scipy.spatial import cKDTree  # type: ignore

import matplotlib.pyplot as plt
import osmnx as ox

Node = object  # Just to avoir VS Code to give Red Flag because Node woudl be undefined.
# We use 'Node' as a generic type for graph node identifiers (e.g., OSM node IDs).

# ============================================================
# 1) Hub scoring from road_id32
# ============================================================


def node_roadid32_score(G, edge_attr: str = "road_id32") -> Dict[Node, int]:
    """
    score[v] = number of distinct road_id32 values incident to node v.
    Works for MultiDiGraph (OSMnx).
    """
    node_to_rids: Dict[Node, Set[object]] = defaultdict(set)

    # OSMnx graphs are typically MultiDiGraph; handle both cases
    if G.is_multigraph():
        edge_iter = G.edges(keys=True, data=True)
        for u, v, k, data in edge_iter:
            rid = data.get(edge_attr, None)
            if rid is None:
                continue
            rids = rid if isinstance(rid, (list, tuple, set)) else [rid]
            for r in rids:
                node_to_rids[u].add(r)
                node_to_rids[v].add(r)
    else:
        edge_iter = G.edges(data=True)
        for u, v, data in edge_iter:
            rid = data.get(edge_attr, None)
            if rid is None:
                continue
            rids = rid if isinstance(rid, (list, tuple, set)) else [rid]
            for r in rids:
                node_to_rids[u].add(r)
                node_to_rids[v].add(r)

    return {n: len(rset) for n, rset in node_to_rids.items()}


def select_candidates(
    G,
    K: int,
    *,
    H_factor: int = 50,
    edge_attr: str = "road_id32",
) -> Tuple[List[Node], np.ndarray, Dict[Node, int]]:
    """
    candidates = top H nodes by hub score, where H = H_factor*K.
    Returns:
        candidates: list of node ids (len=H)
        cand_xy:    (H,2) array of projected coordinates (x,y)
        score:      dict node->hub_score
    """
    score = node_roadid32_score(G, edge_attr=edge_attr)
    ranked = sorted(G.nodes, key=lambda n: score.get(n, 0), reverse=True)

    H = min(len(ranked), max(K, H_factor * K))
    candidates = ranked[:H]
    cand_xy = np.array(
        [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in candidates], dtype=float
    )

    if len(candidates) < K:
        raise ValueError(f"Not enough candidates ({len(candidates)}) for K={K}")

    return candidates, cand_xy, score


# ============================================================
# 2) k-means++ style D^2 sampling on candidates
# ============================================================


def kmeanspp_on_candidates(
    candidates: Sequence[Node],
    cand_xy: np.ndarray,
    K: int,
    *,
    score: Optional[Dict[Node, int]] = None,
    alpha: float = 0.0,
    random_state: int = 0,
    first_seed: str = "max",  # "max" or "random"
    # eps is a safeguard that prevents zero-probability exclusion of low-score nodes. It is only a robustness requirement.
    eps: float = 1e-9,
) -> List[Node]:
    """
    D^2 sampling (k-means++ initialization) restricted to 'candidates'.
    Optionally bias sampling by (score[v]+eps)^alpha.

    first_seed:
      - "max": choose candidate with max score (deterministic)
      - "random": choose candidate weighted by score (if score provided) else uniform
    """
    rng = np.random.default_rng(random_state)
    N = len(candidates)
    if N < K:
        raise ValueError(f"Need at least K candidates, got N={N}, K={K}")

    if score is None:
        cand_score = np.zeros(N, dtype=float)
    else:
        cand_score = np.array([score.get(n, 0) for n in candidates], dtype=float)

    # First seed
    if first_seed == "max" and score is not None:
        i0 = int(np.argmax(cand_score))
    elif first_seed == "random":
        if score is not None and cand_score.sum() > 0:
            probs0 = (cand_score + eps) / (cand_score + eps).sum()
            i0 = int(rng.choice(N, p=probs0))
        else:
            i0 = int(rng.integers(0, N))
    else:
        # If no score, or first_seed=="max" but score not provided
        i0 = int(rng.integers(0, N))

    seeds_idx = [i0]
    seeds0 = [candidates[i0]]

    # Initialize min squared distance to nearest chosen seed
    diff = cand_xy - cand_xy[i0]
    dist2 = np.einsum("ij,ij->i", diff, diff)

    while len(seeds0) < K:
        w = dist2.copy()

        # bias by hub score if requested
        if score is not None and alpha > 0:
            w *= (cand_score + eps) ** alpha

        # exclude already selected
        w[seeds_idx] = 0.0
        total = w.sum()
        if total <= 0:
            # degeneracy: points collapsed or all selected
            remaining = [i for i in range(N) if i not in set(seeds_idx)]
            i_new = int(rng.choice(remaining))
        else:
            probs = w / total
            i_new = int(rng.choice(N, p=probs))

        seeds_idx.append(i_new)
        seeds0.append(candidates[i_new])

        diff = cand_xy - cand_xy[i_new]
        new_dist2 = np.einsum("ij,ij->i", diff, diff)
        dist2 = np.minimum(dist2, new_dist2)

    return seeds0


# ============================================================
# 3) Optional mini-batch refinement of centers
# ============================================================


def minibatch_refine_centers(
    centers_xy: np.ndarray,
    points_xy: np.ndarray,
    *,
    T: int = 2,  # passes
    R: int = 10,  # batches per pass
    B: int = 2000,  # batch size
    random_state: int = 0,
) -> np.ndarray:
    """
    Lightweight mini-batch k-means refinement.
    Updates center positions using random batches without storing full assignments.
    """
    rng = np.random.default_rng(random_state)
    K = centers_xy.shape[0]
    counts = np.zeros(K, dtype=np.int64)

    n_points = points_xy.shape[0]
    if n_points == 0:
        return centers_xy

    for _ in range(T):
        for _ in range(R):
            m = min(B, n_points)
            idx = rng.integers(0, n_points, size=m)
            batch = points_xy[idx]

            # assign each batch point to nearest center
            d2 = ((batch[:, None, :] - centers_xy[None, :, :]) ** 2).sum(axis=2)
            assign = np.argmin(d2, axis=1)

            # update centers incrementally
            for j in range(K):
                mask = assign == j
                if not np.any(mask):
                    continue
                pts = batch[mask]
                counts[j] += pts.shape[0]
                mean_batch = pts.mean(axis=0)
                lr = pts.shape[0] / counts[j]
                centers_xy[j] = centers_xy[j] + lr * (mean_batch - centers_xy[j])

    return centers_xy


def snap_centers_to_nodes(
    centers_xy: np.ndarray, node_ids: Sequence[Node], node_xy: np.ndarray
) -> List[Node]:
    """
    Snap each center to nearest node among node_ids using KDTree on node_xy.
    """
    tree = cKDTree(node_xy)
    _, idx = tree.query(centers_xy, k=1)
    snapped = [node_ids[int(i)] for i in idx]
    return snapped


def deduplicate_and_fill_seeds(
    snapped_nodes: Sequence[Node],
    candidate_nodes: Sequence[Node],
    candidate_xy: np.ndarray,
    *,
    K: int,
    random_state: int = 0,
) -> List[Node]:
    """
    Remove duplicates preserving order; if <K, fill by farthest-first in Euclidean space.
    """
    rng = np.random.default_rng(random_state)

    seen: Set[Node] = set()
    seeds: List[Node] = []
    for n in snapped_nodes:
        if n not in seen:
            seeds.append(n)
            seen.add(n)
        if len(seeds) == K:
            return seeds

    if len(seeds) == 0:
        # fallback: random unique candidates
        idx = rng.choice(len(candidate_nodes), size=K, replace=False)
        return [candidate_nodes[int(i)] for i in idx]

    # Farthest-first fill
    cand_index = {n: i for i, n in enumerate(candidate_nodes)}
    seed_idx = [cand_index[n] for n in seeds if n in cand_index]
    seed_xy = candidate_xy[seed_idx]

    d2_min = np.full(candidate_xy.shape[0], np.inf)
    for sxy in seed_xy:
        d2 = ((candidate_xy - sxy) ** 2).sum(axis=1)
        d2_min = np.minimum(d2_min, d2)

    while len(seeds) < K:
        i = int(np.argmax(d2_min))
        n_new = candidate_nodes[i]
        if n_new in seen:
            d2_min[i] = -1
            continue
        seeds.append(n_new)
        seen.add(n_new)

        sxy = candidate_xy[i]
        d2 = ((candidate_xy - sxy) ** 2).sum(axis=1)
        d2_min = np.minimum(d2_min, d2)

    return seeds


# ============================================================
# 4) One-call function: build seeds
# ============================================================


# I choose the values for a small <100000-node graph; adjust as needed.
def build_seeds(
    G,
    *,
    K: int,
    edge_attr: str = "road_id32",
    H_factor: int = 10,
    alpha: float = 0.25,
    first_seed: str = "max",
    refine: bool = True,
    T: int = 2,
    R: int = 5,
    B: int = 200,
    random_state: int = 0,
) -> List[Node]:
    """
    Full pipeline:
      candidates (top hub nodes) -> D^2 sampling -> optional mini-batch refinement -> snapping -> dedup/fill
    Returns: list of K node ids (seeds)
    """
    candidates, cand_xy, score = select_candidates(
        G, K=K, H_factor=H_factor, edge_attr=edge_attr
    )

    seeds0 = kmeanspp_on_candidates(
        candidates,
        cand_xy,
        K,
        score=score,
        alpha=alpha,
        random_state=random_state,
        first_seed=first_seed,
    )

    if not refine:
        return seeds0

    # seeds0 -> centers_xy
    seed_to_idx = {n: i for i, n in enumerate(candidates)}
    centers_xy = np.array([cand_xy[seed_to_idx[s]] for s in seeds0], dtype=float)

    centers_xy = minibatch_refine_centers(
        centers_xy, cand_xy, T=T, R=R, B=B, random_state=random_state
    )

    snapped = snap_centers_to_nodes(centers_xy, candidates, cand_xy)
    seeds = deduplicate_and_fill_seeds(
        snapped, candidates, cand_xy, K=K, random_state=random_state
    )
    return seeds


# ============================================================
# 5) Hop-bounded BFS partition using provided seeds
# ============================================================


def hop_bounded_bfs_partition(
    G, k: int, *, seeds: Optional[Sequence[Node]] = None, random_state: int = 0
) -> Dict[Node, int]:
    """
    Hop-bounded BFS partition with optional preferred seeds.
    grow blobs of radius k edges, sequentially
    If seeds are provided, they are used as initial seeds when possible, then remaining nodes become seeds.
    """
    rng = np.random.default_rng(random_state)
    unassigned = set(G.nodes)
    P: Dict[Node, int] = {}
    seed_id = 0

    seed_queue = list(seeds) if seeds is not None else []

    def pick_seed(U: Set[Node]) -> Node:
        # Prefer given seeds if present in U
        nonlocal seed_queue
        while seed_queue:
            s = seed_queue.pop(0)
            if s in U:
                return s
        # fallback: random
        return rng.choice(list(U))  # type: ignore

    while unassigned:
        s = pick_seed(unassigned)
        unassigned.remove(s)

        Q = deque([s])
        depth = {s: 0}
        P[s] = seed_id

        while Q:
            v = Q.popleft()
            if depth[v] == k - 1:
                continue
            for w in G.neighbors(v):
                if w in unassigned:
                    unassigned.remove(w)
                    P[w] = seed_id
                    depth[w] = depth[v] + 1
                    Q.append(w)

        seed_id += 1

    return P


# ============================================================
# 6) Graph Voronoi partition using hop distance (each edge has cost 1).
# ============================================================


def graph_voronoi_hops(G, seeds):
    owner = {}
    dist = {}

    Q = deque()

    for s in seeds:
        owner[s] = s
        dist[s] = 0
        Q.append(s)

    while Q:
        v = Q.popleft()
        for w in G.neighbors(v):
            if w not in dist:
                dist[w] = dist[v] + 1
                owner[w] = owner[v]
                Q.append(w)

    return owner, dist


def graph_voronoi_hops_partition(G, seeds):
    """
    True graph Voronoi partition using hop distance (each edge cost = 1).

    Returns
    -------
    P : dict
        P[v] = cell id (0..len(seeds)-1) of the nearest seed by hop distance.
        Tie-break: earlier seed in 'seeds' wins.
    dist : dict
        dist[v] = hop distance to its owning seed.
    """
    seeds = list(seeds)
    seed_to_cid = {s: i for i, s in enumerate(seeds)}

    P = {}
    dist = {}
    Q = deque()

    # initialize multi-source BFS
    for s in seeds:
        P[s] = seed_to_cid[s]
        dist[s] = 0
        Q.append(s)

    # multi-source BFS
    while Q:
        v = Q.popleft()
        for w in G.neighbors(v):
            if w not in dist:  # first time reached => shortest hop distance
                dist[w] = dist[v] + 1
                P[w] = P[v]  # inherits the seed/cell id of v
                Q.append(w)

    return P, dist


# ============================================================
# 7) Plotting utilities: seeds, partition, boundaries
# ============================================================


def plot_seeds(G, seeds: Sequence[Node], *, seed_size: int = 150):
    fig, ax = ox.plot_graph(
        G, node_size=0, edge_color="lightgray", edge_alpha=0.4, show=False, close=False
    )
    xs = [G.nodes[n]["x"] for n in seeds]
    ys = [G.nodes[n]["y"] for n in seeds]
    ax.scatter(
        xs,
        ys,
        s=seed_size,
        facecolors="none",
        edgecolors="red",
        linewidths=2,
        label=f"Seeds ({len(seeds)})",
    )
    ax.legend()
    plt.show()


def plot_partition(
    G,
    P: Dict[Node, int],
    *,
    node_size: int = 8,
    edge_color: str = "lightgray",
    edge_alpha: float = 0.35,
    cmap: str = "tab20",
):
    fig, ax = ox.plot_graph(
        G,
        node_size=0,
        edge_color=edge_color,
        edge_alpha=edge_alpha,
        show=False,
        close=False,
    )

    cell_ids = sorted(set(P.values()))
    num_cells = len(cell_ids)
    colormap = plt.cm.get_cmap(cmap, max(num_cells, 1))

    # group nodes by cell (fast)
    cell_to_nodes: Dict[int, List[Node]] = defaultdict(list)
    for n, cid in P.items():
        cell_to_nodes[cid].append(n)

    for cid, nodes_in_cell in cell_to_nodes.items():
        xs = [G.nodes[n]["x"] for n in nodes_in_cell]
        ys = [G.nodes[n]["y"] for n in nodes_in_cell]
        ax.scatter(xs, ys, s=node_size, color=colormap(cid % colormap.N), alpha=0.7)

    plt.show()


def plot_voronoi_by_seeds(G, seeds, *, edge_alpha=0.4, node_size=12, node_alpha=0.7):
    """
    Plot a discrete (node-based) Voronoi-like partition of graph nodes:
    each node is colored by the nearest seed in Euclidean (x,y).

    Parameters
    ----------
    G : networkx graph (typically OSMnx graph)
        Must have node attributes 'x' and 'y' (projected or lon/lat, but consistent).
    seeds : list
        Node IDs in G used as seeds.
    edge_alpha : float
        Transparency of plotted edges.
    node_size : float
        Size of plotted node dots for regions.
    node_alpha : float
        Transparency of plotted node dots for regions.
    """

    # ---- coordinates for all nodes (in same order as "nodes")
    nodes = list(G.nodes)
    coords = np.array([(G.nodes[n]["x"], G.nodes[n]["y"]) for n in nodes], dtype=float)

    # ---- coordinates for seeds
    seed_coords = np.array(
        [(G.nodes[s]["x"], G.nodes[s]["y"]) for s in seeds], dtype=float
    )

    # ---- nearest seed label for each node (Euclidean)
    labels = cKDTree(seed_coords).query(coords)[
        1
    ]  # index of nearest seed for each node

    # ---- base graph (do not show yet)
    fig, ax = ox.plot_graph(
        G, node_size=0, show=False, close=False, edge_alpha=edge_alpha
    )

    # ---- plot regions as colored nodes
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap=plt.cm.get_cmap("tab20", len(seeds)),
        s=node_size,
        alpha=node_alpha,
        linewidths=0,
        zorder=5,
    )

    # ---- plot seeds on top
    ax.scatter(
        seed_coords[:, 0],
        seed_coords[:, 1],
        s=180,
        facecolors="none",
        edgecolors="black",
        linewidths=2,
        zorder=10,
        label="Seeds",
    )

    ax.set_title("Nearest-seed regions (discrete Voronoi on nodes)")
    plt.show()
    return fig, ax


def compute_boundaries(G, P: Dict[Node, int]) -> Tuple[Set[Node], List[Tuple]]:
    boundary_edges = []
    boundary_nodes: Set[Node] = set()

    if G.is_multigraph():
        for u, v, k in G.edges(keys=True):
            cu = P.get(u)
            cv = P.get(v)
            if cu is None or cv is None:
                continue
            if cu != cv:
                boundary_edges.append((u, v, k))
                boundary_nodes.add(u)
                boundary_nodes.add(v)
    else:
        for u, v in G.edges():
            cu = P.get(u)
            cv = P.get(v)
            if cu is None or cv is None:
                continue
            if cu != cv:
                boundary_edges.append((u, v))
                boundary_nodes.add(u)
                boundary_nodes.add(v)

    return boundary_nodes, boundary_edges


def plot_boundaries(
    G,
    P: Dict[Node, int],
    *,
    edge_color: str = "lightgray",
    edge_alpha: float = 0.35,
    boundary_edge_color: str = "red",
    boundary_edge_alpha: float = 0.9,
    boundary_edge_lw: float = 1.5,
    boundary_node_color: str = "yellow",
    boundary_node_edgecolor: str = "black",
    boundary_node_size: int = 35,
):
    boundary_nodes, boundary_edges = compute_boundaries(G, P)

    fig, ax = ox.plot_graph(
        G,
        node_size=0,
        edge_color=edge_color,
        edge_alpha=edge_alpha,
        show=False,
        close=False,
    )

    # boundary edges drawn as straight segments between node coords
    if G.is_multigraph():
        for u, v, k in boundary_edges:
            x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
            x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=boundary_edge_color,
                alpha=boundary_edge_alpha,
                linewidth=boundary_edge_lw,
            )
    else:
        for u, v in boundary_edges:
            x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
            x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=boundary_edge_color,
                alpha=boundary_edge_alpha,
                linewidth=boundary_edge_lw,
            )

    # boundary nodes
    if boundary_nodes:
        xs = [G.nodes[n]["x"] for n in boundary_nodes]
        ys = [G.nodes[n]["y"] for n in boundary_nodes]
        ax.scatter(
            xs,
            ys,
            s=boundary_node_size,
            c=boundary_node_color,
            edgecolors=boundary_node_edgecolor,
            linewidths=0.8,
            alpha=0.95,
            label=f"Boundary nodes ({len(boundary_nodes)})",
        )
        ax.legend(loc="best")

    plt.show()
    return boundary_nodes, boundary_edges


def plot_graph_voronoi_hops(
    G,
    seeds,
    P,
    *,
    node_size=8,
    node_alpha=0.7,
    edge_color="lightgray",
    edge_alpha=0.35,
    cmap="tab20",
    boundary_edge_color="red",
    boundary_edge_alpha=0.9,
    boundary_edge_lw=1.6,
    seed_size=180,
):
    """
    Plot hop-distance graph Voronoi:
    - edges (base graph)
    - nodes colored by cell
    - boundary edges highlighted
    - seeds drawn on top
    """
    fig, ax = ox.plot_graph(
        G,
        node_size=0,
        edge_color=edge_color,
        edge_alpha=edge_alpha,
        show=False,
        close=False,
    )

    # --- plot colored regions (same idea as your plot_partition) ---
    cell_ids = sorted(set(P.values()))
    colormap = plt.cm.get_cmap(cmap, max(len(cell_ids), 1))

    cell_to_nodes = defaultdict(list)
    for n, cid in P.items():
        cell_to_nodes[cid].append(n)

    for cid, nodes_in_cell in cell_to_nodes.items():
        xs = [G.nodes[n]["x"] for n in nodes_in_cell]
        ys = [G.nodes[n]["y"] for n in nodes_in_cell]
        ax.scatter(
            xs,
            ys,
            s=node_size,
            color=colormap(cid % colormap.N),
            alpha=node_alpha,
            linewidths=0,
            zorder=5,
        )

    # --- boundaries (reuse your compute_boundaries) ---
    boundary_nodes, boundary_edges = compute_boundaries(G, P)

    if G.is_multigraph():
        for u, v, k in boundary_edges:
            x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
            x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=boundary_edge_color,
                alpha=boundary_edge_alpha,
                linewidth=boundary_edge_lw,
                zorder=8,
            )
    else:
        for u, v in boundary_edges:
            x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
            x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=boundary_edge_color,
                alpha=boundary_edge_alpha,
                linewidth=boundary_edge_lw,
                zorder=8,
            )

    # --- seeds on top ---
    seed_coords = [(G.nodes[s]["x"], G.nodes[s]["y"]) for s in seeds if s in G]
    if seed_coords:
        sx, sy = zip(*seed_coords)
        ax.scatter(
            sx,
            sy,
            s=seed_size,
            facecolors="none",
            edgecolors="black",
            linewidths=2,
            zorder=10,
            label=f"Seeds ({len(seed_coords)})",
        )

    ax.set_title("Graph Voronoi by hop distance (regions + boundaries)")
    ax.legend(loc="best")
    plt.show()

    return fig, ax, boundary_nodes, boundary_edges


# ============================================================
# Export
# ============================================================


def export_seeds_txt(path, seeds, node_mapping):
    """
    Export seeds with both original (OSM / NetworkX) node ids
    and corresponding XY node ids.

    Format:
      old_node_id  xy_node_id
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("# old_node_id  xy_node_id\n")
        for old_id in seeds:
            xy_id = node_mapping.get(old_id, -1)
            f.write(f"{old_id} {xy_id}\n")


def export_voronoi_nodes_txt(path, P, dist, node_mapping):
    """
    Export graph Voronoi partition to a text file, including XY node ids.

    Each line:
      old_node_id  xy_node_id  cell_id  hop_distance
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("# old_node_id  xy_node_id  cell_id  hop_distance\n")
        for n, cid in P.items():
            xy_id = node_mapping.get(n, -1)
            hop_d = dist.get(n, -1)
            f.write(f"{n} {xy_id} {cid} {hop_d}\n")


def export_boundary_edges_txt(G, boundary_edges, path, node_mapping):
    """
    Export boundary edges to a text file, including XY node ids.
    """
    with open(path, "w", encoding="utf-8") as f:
        if G.is_multigraph():
            f.write("# u_old  v_old  key  u_xy  v_xy\n")
            for u, v, k in boundary_edges:
                u_xy = node_mapping.get(u, -1)
                v_xy = node_mapping.get(v, -1)
                f.write(f"{u} {v} {k} {u_xy} {v_xy}\n")
        else:
            f.write("# u_old  v_old  u_xy  v_xy\n")
            for u, v in boundary_edges:
                u_xy = node_mapping.get(u, -1)
                v_xy = node_mapping.get(v, -1)
                f.write(f"{u} {v} {u_xy} {v_xy}\n")


def export_boundary_nodes_txt(path, boundary_nodes, node_mapping):
    """
    Export boundary nodes to a text file, with both old and xy node ids.

    Each line:
      old_node_id  xy_node_id
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("# old_node_id  xy_node_id\n")
        # deterministic order for reproducibility
        for old_id in sorted(boundary_nodes):
            xy_id = node_mapping.get(old_id, -1)
            f.write(f"{old_id} {xy_id}\n")


def compute_and_export_boundary_nodes_txt(G, P, path, node_mapping):
    """
    Compute boundary nodes from (G, P) and export them.

    Returns
    -------
    boundary_nodes : set
    """
    boundary_nodes, _ = compute_boundaries(G, P)
    export_boundary_nodes_txt(path, boundary_nodes, node_mapping)
    return boundary_nodes

def export_boundaries_txt(G, P, *, node_mapping, nodes_path, edges_path):
    """
    Compute boundaries and export:
      - boundary nodes: old + xy
      - boundary edges: old + xy (and key if multigraph)
    """
    boundary_nodes, boundary_edges = compute_boundaries(G, P)

    export_boundary_nodes_txt(nodes_path, boundary_nodes, node_mapping)
    export_boundary_edges_txt(G, boundary_edges, edges_path, node_mapping)

    return boundary_nodes, boundary_edges



def export_cells_txt(path, P, node_mapping):
    """
    Export cells as lists of nodes per region, including XY node ids.

    Format:
      cell_id : old_node_id(xy_id) old_node_id(xy_id) ...
    """
    cell_to_nodes = defaultdict(list)
    for n, cid in P.items():
        cell_to_nodes[cid].append(n)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# cell_id : old_node_id(xy_id) ...\n")
        for cid in sorted(cell_to_nodes):
            nodes = " ".join(
                f"{n}({node_mapping.get(n, -1)})" for n in cell_to_nodes[cid]
            )
            f.write(f"{cid} : {nodes}\n")


def export_seeds_latlon_txt(path, G, seeds, node_mapping):
    """
    Export seeds with:
      - original node id
      - xy node id (0..N-1)
      - longitude (x)
      - latitude (y)

    Assumes:
      G.nodes[n]["x"] = longitude
      G.nodes[n]["y"] = latitude
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("# old_node_id  xy_node_id  lon  lat\n")
        for old_id in seeds:
            xy_id = node_mapping.get(old_id, -1)
            x = G.nodes[old_id]["x"]
            y = G.nodes[old_id]["y"]
            f.write(f"{old_id} {xy_id} {x} {y}\n")
