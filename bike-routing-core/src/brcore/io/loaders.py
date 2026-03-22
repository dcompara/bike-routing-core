# loaders.py
# Plain-text loaders for seeds / Voronoi partitions / boundaries
# compatible with the NEW export formats that include BOTH:
#   old_node_id (NetworkX/OSM id)  and  xy_node_id (0..N-1)

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union


# -----------------------------
# helpers

def _iter_data_lines(path: str | Path):
    """Yield non-empty, non-comment lines."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield line


def _pick_col(id_mode: str) -> int:
    """
    For files that store:
      old_node_id  xy_node_id
    return the column index to use.
    """
    if id_mode not in ("old", "xy"):
        raise ValueError("id_mode must be 'old' or 'xy'")
    return 0 if id_mode == "old" else 1


# -----------------------------
# seeds

def load_seeds(path: str | Path, *, id_mode: str = "xy") -> List[int]:
    """
    Load seeds.

    Expected format:
      # old_node_id  xy_node_id
      123456        42
      ...

    Returns a list of node ids (old or xy depending on id_mode).
    """
    col = _pick_col(id_mode)
    out: List[int] = []
    for line in _iter_data_lines(path):
        parts = line.split()
        if len(parts) <= col:
            continue
        out.append(int(parts[col]))
    return out


def load_seeds_latlon(path: str | Path, *, id_mode: str = "xy") -> List[Tuple[int, float, float]]:
    """
    Load seeds with lon/lat.

    Expected format:
      # old_node_id  xy_node_id  lon  lat
      123456        42          2.34 48.85
      ...

    Returns list of tuples:
      (node_id, lon, lat)
    where node_id is chosen by id_mode.
    """
    col = _pick_col(id_mode)
    out: List[Tuple[int, float, float]] = []
    for line in _iter_data_lines(path):
        parts = line.split()
        if len(parts) < 4:
            continue
        node_id = int(parts[col])
        lon = float(parts[2])
        lat = float(parts[3])
        out.append((node_id, lon, lat))
    return out


# -----------------------------
# partition (Voronoi nodes)

def load_partition(path: str | Path, *, id_mode: str = "xy") -> Dict[int, int]:
    """
    Load partition mapping P[node] = cell_id.

    Expected format (new):
      # old_node_id  xy_node_id  cell_id  hop_distance
      123456        42          0        0
      ...

    Returns dict with keys in the chosen id space.
    """
    node_col = _pick_col(id_mode)  # 0=old, 1=xy
    cell_col = 2

    P: Dict[int, int] = {}
    for line in _iter_data_lines(path):
        parts = line.split()
        if len(parts) <= cell_col:
            continue
        node_id = int(parts[node_col])
        cell_id = int(parts[cell_col])
        P[node_id] = cell_id
    return P


def load_partition_with_dist(path: str | Path, *, id_mode: str = "xy") -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Load partition and hop-distance.

    Same expected format as load_partition:
      old  xy  cell  hop

    Returns:
      (P, dist)
    """
    node_col = _pick_col(id_mode)
    cell_col = 2
    dist_col = 3

    P: Dict[int, int] = {}
    dist: Dict[int, int] = {}

    for line in _iter_data_lines(path):
        parts = line.split()
        if len(parts) <= dist_col:
            continue
        node_id = int(parts[node_col])
        P[node_id] = int(parts[cell_col])
        dist[node_id] = int(parts[dist_col])
    return P, dist


# -----------------------------
# boundaries: nodes

def load_boundary_nodes(path: str | Path, *, id_mode: str = "xy") -> Set[int]:
    """
    Load boundary nodes.

    Expected format:
      # old_node_id  xy_node_id
      123456        42
      ...

    Returns a set of node ids (old or xy).
    """
    col = _pick_col(id_mode)
    out: Set[int] = set()
    for line in _iter_data_lines(path):
        parts = line.split()
        if len(parts) <= col:
            continue
        out.add(int(parts[col]))
    return out


# -----------------------------
# boundaries: edges

def load_boundary_edges(
    path: str | Path,
    *,
    id_mode: str = "xy",
    has_key: bool = False,
) -> List[Union[Tuple[int, int], Tuple[int, int, int]]]:
    """
    Load boundary edges.

    Expected formats:

    Simple graph:
      # u_old  v_old  u_xy  v_xy
      123456  123457  42    43

    MultiGraph:
      # u_old  v_old  key  u_xy  v_xy
      123456  123457  0    42    43

    Parameters
    ----------
    id_mode : "xy" or "old"
        Return node ids in that space.
    has_key : bool
        True if the file contains an edge key column.

    Returns
    -------
    If has_key=False:
      List[(u, v)]
    If has_key=True:
      List[(u, v, k)]
    """
    if id_mode not in ("old", "xy"):
        raise ValueError("id_mode must be 'old' or 'xy'")

    out = []
    for line in _iter_data_lines(path):
        parts = line.split()

        if has_key:
            # u_old v_old key u_xy v_xy
            if len(parts) < 5:
                continue
            k = int(parts[2])
            if id_mode == "old":
                u = int(parts[0])
                v = int(parts[1])
            else:
                u = int(parts[3])
                v = int(parts[4])
            out.append((u, v, k))
        else:
            # u_old v_old u_xy v_xy
            if len(parts) < 4:
                continue
            if id_mode == "old":
                u = int(parts[0])
                v = int(parts[1])
            else:
                u = int(parts[2])
                v = int(parts[3])
            out.append((u, v))

    return out


# -----------------------------
# cells (region -> list of nodes)

def load_cells(path: str | Path, *, id_mode: str = "xy") -> Dict[int, List[int]]:
    """
    Load cells (region -> list of nodes).

    Export format we used:
      # cell_id : old_node_id(xy_id) old_node_id(xy_id) ...
      0 : 123456(42) 123457(43) 123458(44)
      1 : 223401(77) 223402(78)

    Returns
    -------
    dict[cell_id] = [node_id, ...] where node_id is chosen by id_mode.
    """
    if id_mode not in ("old", "xy"):
        raise ValueError("id_mode must be 'old' or 'xy'")

    cells: Dict[int, List[int]] = {}

    for line in _iter_data_lines(path):
        # expect: "<cid> : <tokens...>"
        if ":" not in line:
            continue
        left, right = line.split(":", 1)
        cid_str = left.strip()
        cid = int(cid_str)

        tokens = right.strip().split()
        nodes: List[int] = []

        for tok in tokens:
            # tok like "123456(42)"
            if "(" in tok and tok.endswith(")"):
                old_part, xy_part = tok[:-1].split("(", 1)
                if id_mode == "old":
                    nodes.append(int(old_part))
                else:
                    nodes.append(int(xy_part))
            else:
                # fallback: if someone exported only ids
                nodes.append(int(tok))

        cells[cid] = nodes

    return cells
