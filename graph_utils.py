# =============================================================================
# General Functions 
# =============================================================================


import srtm
import networkx as nx
import osmnx as ox

def add_node_elevations(G, google_api_key=None):
    """
    Add node elevations to a graph using the Google Elevation API or SRTM data.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Graph with nodes to which elevation data will be added.
    google_api_key : str, optional
        Google Elevation API key. If provided, the function will use the Google Elevation API.
        If not provided, it will use SRTM data as a fallback.

    Returns
    -------
    G : networkx.MultiDiGraph
        Graph with added elevation data for the nodes.
    """
    if google_api_key:
        try:
            G = ox.elevation.add_node_elevations_google(G, api_key=google_api_key)
            print("Elevation data added using Google Elevation API.")
            return G
        except Exception as e:
            print(f"Failed to use Google Elevation API: {e}. Falling back to SRTM data.")
    
    # Fall back to using SRTM data
    print("Using less accurate SRTM data for elevation.")
    elevation_data = srtm.get_data()

    def add_node_elevations_srtm(G, elevation_data):
        for node, data in G.nodes(data=True):
            elevation = elevation_data.get_elevation(data['y'], data['x'])
            if elevation is not None:
                G.nodes[node]['elevation'] = elevation
            else:
                G.nodes[node]['elevation'] = 0  # Set to 0 if elevation data is not available

    add_node_elevations_srtm(G, elevation_data)
    return G


def is_potential_start_node(G, node, threshold=0.01):
    """
    Determine if a node is a potential starting point for a climb based on elevation.

    A node is considered a potential starting point if it is not adjacent to any
    lower node by more than the specified elevation threshold.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Graph with nodes that have 'elevation' attributes.
    node : int
        The node ID to check.
    threshold : float, optional
        The elevation threshold. If the elevation difference between the node and
        any of its neighbors is greater than this threshold, the node is not a
        potential starting point. Default is 0.01 (1% slope)

    Returns
    -------
    bool
        True if the node is a potential starting point, False otherwise.
    """
    for neighbor in G.neighbors(node):
        if G.nodes[node]['elevation'] - G.nodes[neighbor]['elevation'] > threshold:
            return False
    return True

# Generate all possible climbing paths from a starting node. 
# The paths must continue climbing with a minimum grade for an edge (1% per default) to be considered a climb.
def get_climbing_paths(G, start_node, min_grade=0.01):
    """
    Generate all possible climbing paths from a starting node.

    The paths must continue climbing with a minimum grade for an edge (1% per default) to be considered a climb.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        The input graph.
    start_node : int
        The starting node ID.
    min_grade : float, optional
        The minimum grade required for an edge to be considered part of a climbing path.

    Returns
    -------
    list
        A list of completed climbing paths.
    """

    # Initialize the list of paths with a single path starting from the start_node
    paths = [[start_node]]
    completed_paths = []

    # While there are paths to explore
    while paths:
        # Get the last path in the list to explore its neighbors
        current_path = paths.pop()
        last_node = current_path[-1]

        # Iterate over the neighbors of the last node in the current path
        for neighbor in G.neighbors(last_node):
            # Get the edge data between the last node and the neighbor
            edge_data = G.get_edge_data(last_node, neighbor)
            # Extract the grade of the edge, defaulting to 0 if not available
            grade = edge_data[0].get('grade', 0)
            
            # Check if the edge grade meets the minimum climbing criteria and the neighbor is not already in the path
            if grade >= min_grade and neighbor not in current_path:
                # Create a new path by extending the current path with the neighbor
                new_path = current_path + [neighbor]
                # Add the new path to the list of paths to explore
                paths.append(new_path)

                # Check if the neighbor does not have any higher neighbors (except for the path just created)
                if not any(G.nodes[neighbor]['elevation'] > G.nodes[next_node]['elevation'] for next_node in G.neighbors(neighbor) if next_node not in new_path):
                    # If true, this path has reached its peak and is considered a completed climbing path
                    completed_paths.append(new_path)

    # Return the list of completed climbing paths
    return completed_paths





def filter_paths(G, paths, min_length=100, min_elevation_gain=50):
    """
    Filter paths based on length and minimum elevation gain.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        The input graph.
    paths : list
        A list of paths to be filtered.
    min_length : float, optional
        The minimum length required for a path to be considered valid.
    min_elevation_gain : float, optional
        The minimum elevation gain required for a path to be considered valid.

    Returns
    -------
    list
        A list of valid paths with their length and elevation gain.
    """
    valid_paths = []

    for path in paths:
        path_length = sum(ox.utils_graph.get_route_edge_attributes(G, path, 'length'))
        elevation_gain = G.nodes[path[-1]]['elevation'] - G.nodes[path[0]]['elevation']

        if path_length >= min_length and elevation_gain >= min_elevation_gain:
            valid_paths.append((path, path_length, elevation_gain))
    
    return valid_paths


# ---------------------------------------------------------------------------
# Examples to test the above functions
# ---------------------------------------------------------------------------


def example_add_node_elevations():
    # Example usage of add_node_elevations
    G = ox.graph_from_place("Piedmont, California, USA", network_type='drive')
    google_api_key = "YOUR_GOOGLE_API_KEY_HERE"  # Replace with your actual Google API key
    G = add_node_elevations(G, google_api_key)
    for node, data in G.nodes(data=True):
        print(f"Node {node}: {data}")

def example_find_potential_start_nodes():
    # Example usage of find_potential_start_nodes
    G = nx.MultiDiGraph()
    G.add_nodes_from([(1, {'elevation': 100}), (2, {'elevation': 90}), (3, {'elevation': 110})])
    G.add_edges_from([(1, 2), (2, 3), (1, 3)])
    threshold = 5  # Adjust the threshold as needed
    potential_start_nodes = find_potential_start_nodes(G, threshold)
    print("Potential start nodes:", potential_start_nodes)

if __name__ == "__main__":
    # Choose which example to run
    example_add_node_elevations()
    example_find_potential_start_nodes()
