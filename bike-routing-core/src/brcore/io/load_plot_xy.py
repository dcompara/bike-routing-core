from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from brcore.graph.compact import CompactDiGraph


@dataclass
class XYGraph:
    """
    Full graph loaded from your custom .xy text format:

      header: nodes N edges M attributes K
      nodes:  v id lat_uDeg lon_uDeg elevation_m
      edges:  e u v length hg10 popularity street_width road_id32
    """

    G: CompactDiGraph
    nodes: np.ndarray  # shape (N, 3): [lat_microdeg, lon_microdeg, elev_m]


def load_xy_graph(path: str | Path) -> XYGraph:
    path = Path(path)

    n_nodes: int | None = None
    n_edges: int | None = None

    node_lat: dict[int, int] = {}
    node_lon: dict[int, int] = {}
    node_ele: dict[int, int] = {}

    # Collect edges temporarily
    U: list[int] = []
    V: list[int] = []
    W: list[tuple[float, float, float, float]] = []
    ROAD: list[int] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            tag = parts[0]

            if tag == "nodes":
                # nodes N edges M ...
                try:
                    n_nodes = int(parts[1])
                except Exception as e:
                    raise ValueError(
                        f"Line {line_no}: cannot parse nodes count from: {line!r}"
                    ) from e

                if "edges" in parts:
                    i = parts.index("edges")
                    try:
                        n_edges = int(parts[i + 1])
                    except Exception as e:
                        raise ValueError(
                            f"Line {line_no}: cannot parse edges count from: {line!r}"
                        ) from e
                continue

            if tag == "v":
                # v id lat_microdeg lon_microdeg elevation
                if len(parts) < 5:
                    raise ValueError(f"Line {line_no}: malformed node record: {line!r}")
                vid = int(parts[1])
                lat = int(parts[2])
                lon = int(parts[3])
                ele = int(float(parts[4]))

                node_lat[vid] = lat
                node_lon[vid] = lon
                node_ele[vid] = ele
                continue

            if tag == "e":
                # e u v length hg10 popularity street_width road_id32
                if len(parts) < 8:
                    raise ValueError(f"Line {line_no}: malformed edge record: {line!r}")

                u = int(parts[1])
                v = int(parts[2])
                length = float(parts[3])
                hg10 = float(parts[4])  # already 10x height_gain in the file
                pop = float(parts[5])
                width = float(parts[6])
                road_id32 = int(parts[7])

                U.append(u)
                V.append(v)
                W.append((length, hg10, pop, width))
                ROAD.append(road_id32)
                continue

            raise ValueError(f"Line {line_no}: unknown record type {tag!r}: {line!r}")

    if n_nodes is None or n_edges is None:
        raise RuntimeError("Header line 'nodes ... edges ...' not found or not parsed.")

    if len(U) != n_edges:
        raise RuntimeError(
            f"Expected {n_edges} edges from header, but parsed {len(U)}."
        )

    # Build nodes array
    nodes = np.empty((n_nodes, 3), dtype=np.int32)
    for i in range(n_nodes):
        if i not in node_lat:
            raise RuntimeError(f"Missing node {i} in node section.")
        nodes[i, 0] = node_lat[i]
        nodes[i, 1] = node_lon[i]
        nodes[i, 2] = node_ele[i]

    # Build CSR graph
    u = np.asarray(U, dtype=np.int32)  # source node ids
    v = np.asarray(V, dtype=np.int32)  # target node ids

    # Keep weights as float (length etc.). Convert later if you *really* want fixed-point.
    w = np.asarray(W, dtype=np.float32)  # shape (n_edges, 4)
    road = np.asarray(ROAD, dtype=np.int32)

    counts = np.bincount(u, minlength=n_nodes).astype(np.int32)
    offsets = np.zeros(n_nodes + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts, dtype=np.int32)

    to = np.empty(len(u), dtype=np.int32)
    ww = np.empty((len(u), w.shape[1]), dtype=w.dtype)
    rr = np.empty(len(u), dtype=np.int32)

    cursor = offsets[:-1].copy()
    for i in range(len(u)):
        uu = u[i]
        j = cursor[uu]
        to[j] = v[i]
        ww[j] = w[i]
        rr[j] = road[i]
        cursor[uu] += 1

    G = CompactDiGraph(
        offsets=offsets,
        to=to,
        w=ww,
        road_id=rr,
        n_nodes=n_nodes,
        n_edges=len(u),
        n_obj=ww.shape[1],
    )

    return XYGraph(G=G, nodes=nodes)





def plot_xy_compact_graph(
    xy,
    *,
    seeds=None,
    boundary_nodes=None,
    paths=None,
    edge_alpha=0.25,
    edge_lw=0.6,
    seed_size=80,
    boundary_size=30,
    path_lw=2.5,
):
    """
    Plot an XY CompactDiGraph (CSR format).

    Assumes:
      - xy.G is a CompactDiGraph
      - nodes are indexed 0..N-1
      - xy.nodes[i] -> (x, y, ...)  or dict-like with keys "x","y"
    """

    G = xy.G
    nodes = xy.nodes

    def coord(i):
        p = nodes[i]
        if isinstance(p, dict):
            return float(p["x"]), float(p["y"])
        return float(p[0]), float(p[1])

    fig, ax = plt.subplots()

    # -------------------------
    # Edges (CSR traversal)
    # -------------------------
    for u in range(G.n_nodes):
        to, _, _ = G.neighbors(u)
        if len(to) == 0:
            continue

        x1, y1 = coord(u)
        for v in to:
            x2, y2 = coord(int(v))
            ax.plot(
                [x1, x2],
                [y1, y2],
                linewidth=edge_lw,
                alpha=edge_alpha,
                color="black",
            )

    # -------------------------
    # Boundary nodes
    # -------------------------
    if boundary_nodes is not None and len(boundary_nodes) > 0:
        bx = []
        by = []
        for i in boundary_nodes:
            x, y = coord(i)
            bx.append(x)
            by.append(y)

        ax.scatter(
            bx,
            by,
            s=boundary_size,
            c="yellow",
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
            label=f"Boundary nodes ({len(boundary_nodes)})",
        )

    # -------------------------
    # Seeds
    # -------------------------
    if seeds is not None and len(seeds) > 0:
        sx = []
        sy = []
        for i in seeds:
            x, y = coord(i)
            sx.append(x)
            sy.append(y)

        ax.scatter(
            sx,
            sy,
            s=seed_size,
            facecolors="none",
            edgecolors="red",
            linewidths=1.8,
            zorder=6,
            label=f"Seeds ({len(seeds)})",
        )

    # -------------------------
    # Paths
    # -------------------------
    if paths is not None:
        for path in paths:
            if len(path) < 2:
                continue
            xs = []
            ys = []
            for i in path:
                x, y = coord(i)
                xs.append(x)
                ys.append(y)

            ax.plot(
                xs,
                ys,
                linewidth=path_lw,
                alpha=0.95,
                zorder=10,
            )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("XY Compact graph")
    ax.legend(loc="best")
    plt.show()

    return fig, ax
