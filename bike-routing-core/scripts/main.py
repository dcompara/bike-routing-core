import numpy as np

from brcore.io.load_plot_xy import load_xy_graph,plot_xy_compact_graph
from brcore.io.loaders import load_seeds, load_partition, load_boundary_nodes
from brcore.algo.coords import build_local_xy_int
from brcore.algo.search_space_reduction import search_space_reduction

from brcore.algo import params


def main():
    xy = load_xy_graph("data/graph_Paris_south_4_objectives.xy")
    G = xy.G
    nodes = xy.nodes

    # coords in int meters for fast Euclidean heuristic / corridor
    xy_int = build_local_xy_int(nodes)

    # Example s,t
    s, t = 563, 4424

    # ---- Load in XY id space (0..N-1) ----
    seeds = load_seeds("data/seeds.txt", id_mode="xy")

    # partition file is your NEW voronoi_nodes export:
    #   old  xy  cell  hop
    P = load_partition("data/paris_voronoi_nodes.txt", id_mode="xy")

    # boundary nodes file is your NEW boundary nodes export:
    #   old  xy
    B = load_boundary_nodes("data/paris_voronoi_boundary_nodes.txt", id_mode="xy")

    path = [563, 120, 98, 4424]   # XY ids


    plot_xy_compact_graph(xy, seeds=seeds, boundary_nodes=B, paths=[path])

    # ---- (Optional) demo of Chebyshev heuristic on a dummy vector ----
    # IMPORTANT: x_vec ordering must match params.FEATURES
    x_vec = np.array([9000.0, 120.0, 0.35, 4.2], dtype=float)
    print("Chebyshev score:", params.CHEB.score(x_vec))


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


if __name__ == "__main__":
    main()
