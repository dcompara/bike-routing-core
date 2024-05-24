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

if __name__ == "__main__":
    # Example usage
    # Create a sample graph or load your graph
    G = ox.graph_from_place("Piedmont, California, USA", network_type='drive')

    # Use the function to add elevations
    google_api_key = "YOUR_GOOGLE_API_KEY_HERE"  # Replace with your actual Google API key
    G = add_node_elevations(G, google_api_key)

    # Print the graph nodes with their elevation data to verify
    for node, data in G.nodes(data=True):
        print(f"Node {node}: {data}")
