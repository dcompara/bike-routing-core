import logging

import numpy as np

from brcore.algo import params
from brcore.algo.coords import build_local_xy_int
from brcore.algo.portal_search import (
    ConstraintBox,
    SparsePortalConfig,
    SparsePortalQuery,
    anytime_sparse_portal_search,
    debug_dump_search_state,
)
from brcore.algo.search_space_reduction import search_space_reduction
from brcore.io.load_plot_xy import load_xy_graph
from brcore.io.loaders import (
    load_boundary_edges,
    load_boundary_nodes,
    load_partition,
    load_seeds,
)


logger = logging.getLogger(__name__)


def main() -> None:
    xy = load_xy_graph("data/graph_Paris_south_4_objectives.xy")
    xy_int = build_local_xy_int(xy.nodes)
    seeds = load_seeds("data/seeds.txt", id_mode="xy")
    partition = load_partition("data/paris_voronoi_nodes.txt", id_mode="xy")
    boundary_nodes = load_boundary_nodes(
        "data/paris_voronoi_boundary_nodes.txt",
        id_mode="xy",
    )
    boundary_edges = load_boundary_edges(
        "data/paris_voronoi_boundaries.txt",
        id_mode="xy",
        has_key=True,
    )
    source = boundary_edges[0][0]
    target = boundary_edges[50][1]

    constraints = ConstraintBox.from_bounds(
        lower=np.array([0.0, 0.0, 0.0, 5.0], dtype=float),
        upper=np.array([60_000.0, 5_000.0, 255.0, 40.0], dtype=float),
        weights=params.W,
    )
    kept_cells, kept_nodes = search_space_reduction(
        G=xy.G,
        xy_int=xy_int,
        seeds=seeds,
        partition=partition,
        boundary_nodes=boundary_nodes,
        s=source,
        t=target,
        corridor_slack_m=1500,
        max_hops_from_boundary=1,
    )
    logger.info(
        "stage_1 kept_cells=%s kept_nodes=%s",
        len(kept_cells),
        len(kept_nodes),
    )

    query = SparsePortalQuery(
        source=source,
        target=target,
        constraints=constraints,
        time_budget_s=params.DEFAULT_TIME_BUDGET_S,
        archive_size=params.DEFAULT_ARCHIVE_SIZE,
    )
    config = SparsePortalConfig(
        max_active_portals_per_cell=params.MAX_ACTIVE_PORTALS_PER_CELL,
        max_labels_per_portal=params.MAX_LABELS_PER_PORTAL,
        max_shortcuts_per_pair=params.MAX_SHORTCUTS_PER_PAIR,
        local_expand_limit=params.LOCAL_EXPAND_LIMIT,
        advance_round_budget=params.ADVANCE_ROUND_BUDGET,
        trace_search=True,
        trace_portals=None,
        trace_cells=None,
        max_trace_events=300,
    )
    state = anytime_sparse_portal_search(
        G=xy.G,
        partition=partition,
        boundary_nodes=boundary_nodes,
        kept_nodes=kept_nodes,
        query=query,
        config=config,
    )
    debug_dump_search_state(state, query)
    logger.info(
        "search_complete archive_size=%s trace_events=%s",
        len(state.archive.entries),
        state.trace_event_count,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    main()
