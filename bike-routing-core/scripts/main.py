import numpy as np
import logging

from brcore.io.load_plot_xy import load_xy_graph, plot_xy_compact_graph
from brcore.io.loaders import (
    load_boundary_edges,
    load_boundary_nodes,
    load_partition,
    load_seeds,
)
from brcore.algo.coords import build_local_xy_int
from brcore.algo.portal_search import (
    ConstraintBox,
    SparsePortalConfig,
    SparsePortalQuery,
    anytime_sparse_portal_search,
)
from brcore.algo.search_space_reduction import search_space_reduction

from brcore.algo import params


def main(show_plot: bool = False):
    xy = load_xy_graph("data/graph_Paris_south_4_objectives.xy")
    G = xy.G
    nodes = xy.nodes

    # coords in int meters for fast Euclidean heuristic / corridor
    xy_int = build_local_xy_int(nodes)

    # ---- Load in XY id space (0..N-1) ----
    seeds = load_seeds("data/seeds.txt", id_mode="xy")

    # partition file is your NEW voronoi_nodes export:
    #   old  xy  cell  hop
    P = load_partition("data/paris_voronoi_nodes.txt", id_mode="xy")

    # boundary nodes file is your NEW boundary nodes export:
    #   old  xy
    B = load_boundary_nodes("data/paris_voronoi_boundary_nodes.txt", id_mode="xy")

    # Minimal portal demo pair:
    # choose a real inter-cell boundary edge so the first thin skeleton
    # already exercises portal-to-portal joining.
    boundary_edges = load_boundary_edges(
        "data/paris_voronoi_boundaries.txt",
        id_mode="xy",
        has_key=True,
    )
    # s, t, _ = boundary_edges[0] #Simple test
    s = boundary_edges[0][0]
    t = boundary_edges[50][1]

    if show_plot:
        path = [s, t]  # direct boundary-edge demo
        plot_xy_compact_graph(xy, seeds=seeds, boundary_nodes=B, paths=[path])

    # Demo box: intentionally broad for the first thin skeleton.
    box = ConstraintBox.from_bounds(
        lower=np.array([0.0, 0.0, 0.0, 5.0], dtype=float),
        upper=np.array([60_000.0, 5_000.0, 255.0, 40.0], dtype=float),
        weights=params.W,
    )
    kept_cells, kept_nodes = search_space_reduction(
        G=G,
        xy_int=xy_int,
        seeds=seeds,
        partition=P,
        boundary_nodes=B,
        s=s,
        t=t,
        corridor_slack_m=1500,  # tune later
        max_hops_from_boundary=1,  # tune later
    )

    print("Stage-1 reduction:")
    print("  kept cells:", len(kept_cells))
    print("  kept nodes:", len(kept_nodes))

    query = SparsePortalQuery(
        source=s,
        target=t,
        constraints=box,
        time_budget_s=params.DEFAULT_TIME_BUDGET_S,
        archive_size=params.DEFAULT_ARCHIVE_SIZE,
    )
    config = SparsePortalConfig(
        max_active_portals_per_cell=params.MAX_ACTIVE_PORTALS_PER_CELL,
        max_labels_per_portal=params.MAX_LABELS_PER_PORTAL,
        max_shortcuts_per_pair=params.MAX_SHORTCUTS_PER_PAIR,
        local_expand_limit=params.LOCAL_EXPAND_LIMIT,
        advance_round_budget=params.ADVANCE_ROUND_BUDGET,
    )

    state = anytime_sparse_portal_search(
        G=G,
        partition=P,
        boundary_nodes=B,
        kept_nodes=kept_nodes,
        query=query,
        config=config,
    )

    print("Portal skeleton:")
    print("  active portals:", len(state.active_portals))
    print("  overlay nodes with out-edges:", len(state.overlay.out_edges))
    print("  local engines touched:", len(state.local_engines))
    print("  archive size:", len(state.archive.entries))

    for idx, entry in enumerate(state.archive.entries, start=1):
        route_vec = entry.metrics.route_vector()
        print(
            f"  route {idx}: score={entry.score:.4f} "
            f"length={route_vec[0]:.1f} elev={route_vec[1]:.1f} "
            f"avg_pop={route_vec[2]:.2f} avg_width={route_vec[3]:.2f} "
            f"path_nodes={len(entry.path_nodes)}"
        )

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s"
    )
    main()
