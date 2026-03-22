import osmnx as ox

from .partition import (
    build_seeds,
    graph_voronoi_hops_partition,
    compute_boundaries,
    plot_graph_voronoi_hops,
    export_seeds_txt,
    export_boundary_nodes_txt,
    export_seeds_latlon_txt,
    export_voronoi_nodes_txt,
    export_boundary_edges_txt,
    export_cells_txt,
)

from .save_graph_xy import export_graph_to_xy




G = ox.io.load_graphml(
    "./data/South_graph_with_road_ids.graphml",
    node_dtypes=None,
    edge_dtypes=None,
    graph_dtypes=None,
)


# IF we want to plot the graph
# ox.plot_graph(G)

# -----------------------------
# Node mapping (new node IDs -> 0..N-1)

# road_id32[(label, highway)] -> road_id32
# node_mapping[old_node_id] -> xy_node_id (0..N-1)

node_mapping, road_id32  = export_graph_to_xy(G, "./data/graph_Paris_south_4_objectives.xy") # type: ignore

# ------------------------------------------------
# ------------- Partitioning process -------------
# ------------------------------------------------

# --- Build seeds ---
K = 100  # Number of seeds
seeds = build_seeds(G, K=K, H_factor=10, alpha=0.25, refine=True)
# plot_seeds(G, seeds)
# plot_voronoi_by_seeds(G, seeds)

# --- Export seeds (new node IDs) ---
export_seeds_txt("./data/seeds.txt", seeds, node_mapping)

export_seeds_latlon_txt("./data/seeds_latlon.txt", G, seeds, node_mapping)


# --- Caculate the Partiton Voronoi hop + save it ---

P, dist = graph_voronoi_hops_partition(G, seeds)
boundary_nodes, boundary_edges = compute_boundaries(G, P)
plot_graph_voronoi_hops(G, seeds, P)
# --- Export partition (new node IDs) ---
prefix = "./data/paris_voronoi"
export_voronoi_nodes_txt(prefix + "_nodes.txt", P, dist, node_mapping)
export_boundary_edges_txt(G, boundary_edges, prefix + "_boundaries.txt", node_mapping)
export_cells_txt(prefix + "_cells.txt", P, node_mapping)
export_boundary_nodes_txt(prefix + "_boundary_nodes.txt", boundary_nodes, node_mapping)




exit(0) # stop here for now
"""
# --- Build partition ---
k = 6  # Number of partitions
P = hop_bounded_bfs_partition(G, k=k, seeds=seeds)
plot_partition(G, P)


boundary_nodes, boundary_edges = compute_boundaries(G, P)


# --- Export partition (new node IDs) ---
with open("./data/partition.txt", "w") as f:
    for osm_node, cell_id in P.items():
        new_node = node_mapping[osm_node]
        f.write(f"{new_node} {cell_id}\n")

# --- Export boundary nodes (new node IDs) ---
with open("./data/boundary_nodes.txt", "w") as f:
    for osm_node in boundary_nodes:
        new_node = node_mapping[osm_node]
        f.write(f"{new_node}\n")

# Assuming G is your graph and node_mapping is already created
node_mapping = {old_id: new_id for new_id, old_id in enumerate(G.nodes())}

# Create the reverse mapping (new_id to old_id)
reverse_mapping = {new_id: old_id for old_id, new_id in node_mapping.items()}

# Save old_to_new mapping to a text file
with open("./data/old_to_new_mapping.txt", "w") as f:
    for old_id, new_id in node_mapping.items():
        f.write(f"{old_id}\t{new_id}\n")  # Using tab as a separator

# Save new_to_old mapping to a text file
with open("./data/new_to_old_mapping.txt", "w") as f:
    for new_id, old_id in reverse_mapping.items():
        f.write(f"{new_id}\t{old_id}\n")  # Using tab as a separator


print("Exported seeds, partition, and boundary nodes using new node IDs.")
"""
