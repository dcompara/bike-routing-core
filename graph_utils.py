# =============================================================================
# General Functions 
# =============================================================================

import numpy as np
import pandas as pd
import srtm
import networkx as nx
import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString
from matplotlib.colors import Normalize, to_hex
from matplotlib.cm import get_cmap
import folium
import collections

import networkx as nx
from typing import List



import networkx as nx
import osmnx as ox
import pandas as pd

def calculate_total_attributes_multiplekm(G, route_node_list, attribute='length', positive_only=False):
    """
    Calculate the total value of a specified attribute for the route and the number of kilometers taken multiple times.

    Parameters:
    G (networkx.MultiDiGraph): The graph containing nodes and edges.
    route_node_list (list): List of node IDs constituting the path.
    attribute (str): The attribute to calculate ('length', 'height_gain', etc.).
    positive_only (bool): If True, sum only the positive values of the attribute.

    Returns:
    tuple: Total value of the specified attribute for the route, and the number of kilometers taken multiple times.
    """
    gdf_edges = ox.routing.route_to_gdf(G, route_node_list)

     # Ensure the attribute is numeric
    gdf_edges[attribute] = pd.to_numeric(gdf_edges[attribute], errors='coerce')
    
    total_value = 0
    duplicate_distance = 0
    edge_count = {}

    for u, v in zip(route_node_list[:-1], route_node_list[1:]):
        edge_data = G.get_edge_data(u, v)
        
        for key in edge_data:
            edge = edge_data[key]
            value = edge.get(attribute, 0)
            edge_id = (u, v, key)

            if positive_only and value <= 0:
                continue

            total_value += value

            if edge_id in edge_count:
                edge_count[edge_id] += 1
                duplicate_distance += value
            else:
                edge_count[edge_id] = 1

    return total_value, duplicate_distance

# Example usage
# G = ox.graph_from_place('Piedmont, California, USA', network_type='bike')
# route_node_list = [starting_node, intermediate_node, destination_node]
# total_length, duplicate_length = calculate_total_attributes(G, route_node_list, attribute='length')
# print(f"Total Length: {total_length} km, Duplicate Length: {duplicate_length} km")








def calculate_total_attributes(G, route_node_list, attribute = 'length', positive_only=False):
    """
    Calculate the total value of a specified attribute for the route.

    Parameters:
    G (networkx.MultiDiGraph): The graph containing nodes and edges.
    route_node_list (list): List of node IDs constituting the path.
    attribute (str): The attribute to calculate ('length', 'height_gain', etc.).
    positive_only (bool): If True, sum only the positive values of the attribute.

    Returns:
    float: Total value of the specified attribute for the route.
    """
    gdf_edges = ox.routing.route_to_gdf(G, route_node_list)

    # Ensure the attribute is numeric
    gdf_edges[attribute] = pd.to_numeric(gdf_edges[attribute], errors='coerce')

    if positive_only:
        return float(gdf_edges[gdf_edges[attribute] > 0][attribute].sum())
    else:
        return float(gdf_edges[attribute].sum())


def calculate_route(G, nodes, weight='length'):
    """
    Calculate the complete route passing through the given waypoints.

    Parameters:
    G (networkx.MultiDiGraph): The graph containing nodes and edges.
    nodes (list): List of nodes.
    weight (str): The attribute to calculate ('length', 'height_gain', etc.).

    Returns:
    list: List of node IDs constituting the complete route.
    """

    route = []
    for i in range(len(nodes) - 1):
        try:
            segment = nx.shortest_path(G, nodes[i], nodes[i + 1], weight=weight)
            route.extend(segment[:-1])  # Exclude the last node to avoid duplication
        except nx.NetworkXNoPath:
            # print(f"No path between {nodes[i]} and {nodes[i + 1]}")
            return []
    route.append(nodes[-1])  # Append the last waypoint to complete the route
    return route





def get_d_plus_total(G, route_node_list):
    """
    Calculate the total positive elevation gain for a given list of nodes in a graph.

    Parameters:
    G (networkx.Graph): The graph containing nodes with elevation data.
    route_node_list (list): List of node IDs representing the path.

    Returns:
    float: Total positive elevation gain in meters.
    """
    total_gain = 0

    for i in range(len(route_node_list) - 1):
        elevation_current = G.nodes[route_node_list[i]].get('elevation', 0)
        elevation_next = G.nodes[route_node_list[i + 1]].get('elevation', 0)
        gain = elevation_next - elevation_current

        if gain > 0:  # Only consider elevation gain, ignore drops
            total_gain += gain

    return total_gain


def get_d_plus(gdf_route):
    """
    Calculate the total positive elevation gain from a GeoDataFrame of route nodes.

    Parameters:
    gdf_route (gpd.GeoDataFrame): GeoDataFrame containing route nodes with elevation data.

    Returns:
    float: Total positive elevation gain in meters.
    """
    # Extract the list of elevations from the GeoDataFrame
    elev_lst = list(gdf_route['elevation'])
    # Initialize the total positive elevation gain
    d_plus_out = 0
    
    # Iterate through the elevation list, starting from the second element
    for i, val in enumerate(elev_lst[1:], start=1):
        # If the current elevation is higher than the previous one, add the difference to the total positive elevation gain
        if val > elev_lst[i - 1]:
            d_plus_out += val - elev_lst[i - 1]
        # Debug print statement (commented out)
        # print(f'{i} - previous elevation: {elev_lst[i - 1]}, current elevation: {val}, d+: {d_plus_out}')
    
    return d_plus_out






def filter_by_highway_types(G: nx.MultiDiGraph, allowed_types: List[str]) -> nx.MultiDiGraph:
    """
    Filters the edges of the graph G by allowed highway types.

    Parameters:
    G (nx.MultiDiGraph): The input graph.
    allowed_types (List[str]): A list of allowed highway types.

    Returns:
    nx.MultiDiGraph: The filtered graph with only the allowed highway types.
    """
    edges_to_remove = []
    for u, v, k, data in G.edges(keys=True, data=True):
        highway_type = data.get("highway")
        if isinstance(highway_type, list):
            # If highway is a list, check if any type in the list is allowed
            if not any(ht in allowed_types for ht in highway_type):
                edges_to_remove.append((u, v, k))
        else:
            if highway_type not in allowed_types:
                edges_to_remove.append((u, v, k))
    G.remove_edges_from(edges_to_remove)
    return G



def filter_by_popularity(G: nx.MultiDiGraph, min_popularity: float) -> nx.MultiDiGraph:
    """
    Filters the edges of the graph G by a minimum popularity threshold.

    Parameters:
    G (nx.MultiDiGraph): The input graph.
    min_popularity (float): The minimum popularity threshold. Edges with popularity less than or equal to this value will be removed.

    Returns:
    nx.MultiDiGraph: The filtered graph with edges having popularity greater than the given threshold.
    """
    edges_to_remove = []
    for u, v, k, data in G.edges(keys=True, data=True):
        popularity = float(data.get("popularity", 0))
        if popularity <= min_popularity:
            edges_to_remove.append((u, v, k))
    G.remove_edges_from(edges_to_remove)
    return G


'''


FUNCTIO FROM THE BAYESIAN OPTIMIZATION. MAY BE OBSOLTES

'''




def get_loss(route, gdf_nodes_route, kms_target=30, verbose=True):
    """
    Compute the loss of a route compared to target parameters.

    The loss is calculated based on three factors:
    1. Distance deviation from the target distance.
    2. Positive elevation gain (d+).
    3. Route discovery (penalty for repeating streets/roads).

    Parameters:
    route (list): List of nodes in the route.
    gdf_nodes_route (gpd.GeoDataFrame): GeoDataFrame containing route nodes with cumulative distance and elevation data.
    kms_target (float, optional): Target distance for the route in kilometers. Default is 30.
    verbose (bool, optional): If True, prints detailed loss information. Default is True.

    Returns:
    float: Total loss value.
    """
    # 1 - Calculate loss based on distance deviation from the target
    loss_max = 1000

    gdf_nodes_route = gpd.GeoDataFrame(gdf_nodes_route)


    if gdf_nodes_route.empty:
        return loss_max  
        

    route_dist = gdf_nodes_route.iloc[-1]['cum_dist']  # Loop distance (kms)
    dist_delta = abs(kms_target - route_dist)
    tolerance_target = kms_target * 0.50

    if dist_delta < tolerance_target:
        loss_dist = 0
    else:
        loss_dist = dist_delta

    # 2 - Calculate loss based on positive elevation gain (d+)
    d_plus = get_d_plus(gdf_nodes_route)  # d+ [positive elevation] (m)
    # loss_dplus = (1 / d_plus) * 10000 if d_plus != 0 else float('inf')
    loss_dplus = 100 * route_dist / d_plus

    # 3 - Calculate loss based on route discovery (penalty for repeating streets/roads)
    unique_numbers = [el for el, cnt in collections.Counter(route).items() if cnt == 1]
    if len(unique_numbers) == 0:
        loss_discovery = float('inf')
    else:
        # loss_discovery = len(route) / len(unique_numbers) * 10
        loss_discovery = len(route) / len(unique_numbers) * 3

    # Total loss
    loss = loss_dist + loss_dplus + loss_discovery


    # if verbose:
    print(f'\n--------- loss: {loss:.2f} ------ loss kms: {loss_dist:.2f} ({route_dist:.2f}), d+: {loss_dplus:.2f} ({d_plus}), ' f'twice_penalty: {loss_discovery:.2f} ------')

    return loss



def random_gps_waypoints(n=10, gps_y_min=-180, gps_y_max=180, gps_x_min=-90, gps_x_max=90):
    """
    Create n random GPS waypoints.

    Parameters:
    n (int, optional): Number of waypoints to generate. Default is 10.
    gps_y_min (float, optional): Minimum latitude value. Default is -180.
    gps_y_max (float, optional): Maximum latitude value. Default is 180.
    gps_x_min (float, optional): Minimum longitude value. Default is -90.
    gps_x_max (float, optional): Maximum longitude value. Default is 90.

    Returns:
    list of tuple: List of randomly generated GPS waypoints as (latitude, longitude).
    """
    return [
        (
            round(np.random.uniform(low=gps_y_min, high=gps_y_max), 7),
            round(np.random.uniform(low=gps_x_min, high=gps_x_max), 7)
        )
        for _ in range(n)
    ]

import numpy as np
import random


# TO BE CHECKED
def random_gps_waypoints_from_list(n, waypoints, radius_meters, gps_y_min, gps_y_max, gps_x_min, gps_x_max):
    """
    Choose n random GPS waypoints from a provided list, with a random point around each waypoint within a given radius in meters.

    Parameters:
    n (int): Number of waypoints to generate.
    waypoints (list of tuple): List of GPS waypoints as (latitude, longitude).
    radius_meters (float): Radius around each waypoint in meters.
    gps_y_min (float): Minimum latitude value.
    gps_y_max (float): Maximum latitude value.
    gps_x_min (float): Minimum longitude value.
    gps_x_max (float): Maximum longitude value.

    Returns:
    list of tuple: List of randomly generated GPS waypoints as (latitude, longitude).
    """
    def meters_to_degrees(meters, latitude):
        # 1 degree of latitude is approximately 111320 meters
        lat_degrees = meters / 111320
        
        # 1 degree of longitude is approximately 111320 * cos(latitude) meters
        lon_degrees = meters / (111320 * np.cos(np.radians(latitude)))
        
        return lat_degrees, lon_degrees
    
    def generate_point_around(lat, lon, radius_meters):
        lat_degrees, lon_degrees = meters_to_degrees(radius_meters, lat)
        
        while True:
            # Random distance and angle
            distance = np.random.uniform(0, 1)  # Fractional distance
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Offset in lat/lon degrees
            delta_lat = distance * lat_degrees * np.cos(angle)
            delta_lon = distance * lon_degrees * np.sin(angle)
            
            new_lat = lat + delta_lat
            new_lon = lon + delta_lon
            
            if gps_y_min <= new_lat <= gps_y_max and gps_x_min <= new_lon <= gps_x_max:
                return round(new_lat, 7), round(new_lon, 7)

    if n > len(waypoints):
        raise ValueError("Number of waypoints to generate cannot be greater than the number of provided waypoints.")

    chosen_waypoints = random.sample(waypoints, n)
    
    return [generate_point_around(lat, lon, radius_meters) for lat, lon in chosen_waypoints]




def create_one_route(G, gdf_nodes, gdf_edges, start_point, end_point, attribute = 'length'):
    """
    Compute the shortest path for a given starting and ending point.

    Parameters:
    G (networkx.Graph): The graph containing nodes and edges.
    gdf_nodes (gpd.GeoDataFrame): GeoDataFrame containing the nodes of the graph.
    gdf_edges (gpd.GeoDataFrame): GeoDataFrame containing the edges of the graph.
    start_point (tuple): Starting point as (latitude, longitude).
    end_point (tuple): Ending point as (latitude, longitude).
    attribute (str): The attribute to calculate ('length', 'elevation', etc.).

    Returns:
    tuple: A tuple containing the route (list of node IDs) and the GeoDataFrame of the route nodes.
    """
    # Find the nearest nodes in the graph to the start and end points
    orig = ox.distance.nearest_nodes(G, X=start_point[1], Y=start_point[0])
    dest = ox.distance.nearest_nodes(G, X=end_point[1], Y=end_point[0])
    
    # Compute the shortest path
    try:
        route = nx.shortest_path(G, orig, dest, weight="attribute")
    except nx.NetworkXNoPath:
        route = []
    if not route or len(route) == 1:
        return [], gpd.GeoDataFrame()
    
    # Subset of gdf_nodes for the route (list of osmid) only
    # gdf_nodes_route = gdf_nodes.loc[route]
    gdf_nodes_route =  gpd.GeoDataFrame(gdf_nodes.loc[route])

    # Add elevation, cumulative distance, and highway type into the gdf_nodes_route
    route_elevations = []
    route_dist = [0]
    route_highways = ['unclassified']
    elevation = 0
    dist = 0
    n = 0
    
    for row in gdf_nodes_route.iterrows():
        elevation = row[1]['elevation']
        route_elevations.append(elevation)
        if n != 0 and n < len(route) - 1:
            dist = gdf_edges.xs((row[0], route[n+1]), level=('u', 'v'))['length'].values[0]
            highway = gdf_edges.xs((row[0], route[n+1]), level=('u', 'v'))['highway'].values[0]
            route_dist.append(dist)
            route_highways.append(highway)
        n += 1
    
    gdf_nodes_route['elevation'] = route_elevations
    gdf_nodes_route['cum_dist'] = np.cumsum(route_dist + [dist]) / 1000
    gdf_nodes_route['highway'] = route_highways + ['unclassified']
    
    return route, gdf_nodes_route


def generate_route(G, gdf_nodes, gdf_edges, start, waypoints, end):
    """
    Create a  route starting from a point, passing through a list of waypoints ad ending to the a last point

    Parameters:
    G (networkx.Graph): The graph containing nodes and edges.
    gdf_nodes (gpd.GeoDataFrame): GeoDataFrame containing the nodes of the graph.
    gdf_edges (gpd.GeoDataFrame): GeoDataFrame containing the edges of the graph.
    start (tuple): Starting point as (latitude, longitude).
    waypoints (list of tuple): List of waypoints as (latitude, longitude).

    Returns:
    tuple: A tuple containing the final route (list of node IDs), the GeoDataFrame of the route nodes, and a list of individual routes.
    """
    routes_list = []
    points = [start] + waypoints + [end]
    full_route = []
    combined_gdf_nodes_route = pd.DataFrame()
    
    for i in range(len(points) - 1):
        route, gdf_nodes_route = create_one_route(G, gdf_nodes, gdf_edges, points[i], points[i+1])
        if not route or gdf_nodes_route.empty:
            return -1, [], routes_list
        
        # Extend the full route by appending the new route segment (excluding the last node to avoid duplication)
        full_route = full_route[:-1] + route if full_route else route
        routes_list.append(route)
        
        if combined_gdf_nodes_route.empty:
            combined_gdf_nodes_route = gdf_nodes_route
        else:
            last_cum_dist = combined_gdf_nodes_route.iloc[-1]['cum_dist']
            gdf_nodes_route['cum_dist'] += last_cum_dist
            combined_gdf_nodes_route = pd.concat(
                [combined_gdf_nodes_route[:-1].reset_index(drop=True), 
                 gdf_nodes_route[:-1].reset_index(drop=True)], axis=0)
    
    return full_route, combined_gdf_nodes_route, routes_list





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
        
            # Check if edge_data exists and has the expected structure 
            if edge_data and 0 in edge_data:
                # Extract the grade of the edge, defaulting to 0 if not available
                grade = edge_data[0].get('grade', 0)
            else:
                grade = 0  # Default to 0 if edge_data is None or does not have the expected structure
            
            
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


import geopandas as gpd
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_hex
from matplotlib.cm import get_cmap

def convert_paths_to_gdf_with_grades(G, paths):
    """
    Convert a list of paths (each a list of nodes) to a GeoDataFrame, including grades.

    Parameters:
    G (networkx.Graph): The graph containing nodes and edges.
    paths (list of lists): A list of paths, where each path is a list of node IDs.

    Returns:
    gpd.GeoDataFrame: GeoDataFrame containing geometries and grades.
    """
    lines = []
    grades = []

    for path in paths:
        coords = []
        path_grades = []

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = G.get_edge_data(u, v)
            
            for key, data in edge_data.items():
                if 'geometry' in data:
                    coords.extend(list(data['geometry'].coords))
                    path_grades.extend([data.get('grade', 0)] * len(data['geometry'].coords))
                else:
                    coords.append((G.nodes[u]['x'], G.nodes[u]['y']))
                    coords.append((G.nodes[v]['x'], G.nodes[v]['y']))
                    path_grades.append(data.get('grade', 0))
                    path_grades.append(data.get('grade', 0))
        
        line = LineString(coords)
        lines.append(line)
        grades.append(path_grades)

    gdf = gpd.GeoDataFrame({'geometry': lines, 'grades': grades})
    return gdf

def get_color_for_grade(grade, cmap='plasma'):
    """
    Get a color for a given grade using a color map.

    Parameters:
    grade (float): The grade value to color.
    cmap (str): The colormap name to use.

    Returns:
    str: The hexadecimal color code.
    """
    norm = Normalize(vmin=0, vmax=0.15)  # Adjust vmax based on the expected grade range
    cmap = get_cmap(cmap)
    color = cmap(norm(grade))
    return to_hex(color)


def convert_gdf_to_gpx(gdf, name):
    trace_points = "\n".join([
        TRACE_POINT.format(
            lat=row['y'],
            lon=row['x'],
            elevation=row['elevation'],
            timestamp=datetime.now().isoformat()
        ) for idx, row in gdf.iterrows()
    ])

    gpx_payload = TEMPLATE.format(
        name=name,
        timestamp=datetime.now().isoformat(),
        trace_points=trace_points
    )

    return gpx_payload

def convert_paths_to_gdf_with_grades(G, paths):
    """
    Convert a list of paths (each a list of nodes) to a GeoDataFrame, including grades.

    Parameters:
    G (networkx.Graph): The graph containing nodes and edges.
    paths (list of lists): A list of paths, where each path is a list of node IDs.

    Returns:
    gpd.GeoDataFrame: GeoDataFrame containing geometries and grades.
    """
    lines = []
    grades = []

    for path in paths:
        coords = []
        path_grades = []

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = G.get_edge_data(u, v)
            
            for key, data in edge_data.items():
                if 'geometry' in data:
                    coords.extend(list(data['geometry'].coords))
                    path_grades.extend([data.get('grade', 0)] * len(data['geometry'].coords))
                else:
                    coords.append((G.nodes[u]['x'], G.nodes[u]['y']))
                    coords.append((G.nodes[v]['x'], G.nodes[v]['y']))
                    path_grades.append(data.get('grade', 0))
                    path_grades.append(data.get('grade', 0))
        
        line = LineString(coords)
        lines.append(line)
        grades.append(path_grades)

    gdf = gpd.GeoDataFrame({'geometry': lines, 'grades': grades})
    return gdf

def get_color_for_grade(grade, cmap='plasma'):
    """
    Get a color for a given grade using a color map.

    Parameters:
    grade (float): The grade value to color.
    cmap (str): The colormap name to use.

    Returns:
    str: The hexadecimal color code.
    """
    norm = Normalize(vmin=0, vmax=0.15)  # Adjust vmax based on the expected grade range
    cmap = get_cmap(cmap)
    color = cmap(norm(grade))
    return to_hex(color)


def display_paths_on_map(G, paths, cmap='plasma'):
    """
    Display the paths on an OpenStreetMap with color coding based on grades.

    Parameters:
    G (networkx.Graph): The graph containing nodes and edges.
    paths (list of lists): A list of paths, where each path is a list of node IDs.
    cmap (str): The colormap name to use for grading.

    Returns:
    folium.Map: A Folium map with the paths displayed.
    """
    # Convert paths to GeoDataFrame
    gdf_paths = convert_paths_to_gdf_with_grades(G, paths)
    
    # Create a Folium map centered on the area of interest
    centroid = gdf_paths.geometry.unary_union.centroid
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=14, tiles='cartodbpositron')

    # Add each path to the Folium map
    for _, row in gdf_paths.iterrows():
        if row.geometry.geom_type == "LineString":
            coords = [(pt[1], pt[0]) for pt in row.geometry.coords]
            grades = row.grades
            for i in range(len(coords) - 1):
                segment = coords[i:i+2]
                grade = grades[i]
                color = get_color_for_grade(grade, cmap)
                folium.PolyLine(
                    locations=segment,
                    color=color, weight=2.5
                ).add_to(m)

    return m


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
