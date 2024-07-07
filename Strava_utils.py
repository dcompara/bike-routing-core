# Strava_utils.py

# ============================================================================================
# General Functions useful for importing and treat Strava HeatMap with Open Street map
# ============================================================================================

import requests
from PIL import Image
import numpy as np
import io
import networkx as nx

import importlib
import keys
from getpass import getpass
from stravacookies import StravaCookieFetcher


def fetch_strava_cookies(email, password):
    """
    Fetch Strava cookies required for authentication.

    Parameters
    ----------
    email : str
        The email address used for Strava login.
    password : str
        The password used for Strava login.

    Returns
    -------
    tuple
        A tuple containing the CloudFront key pair ID, policy, and signature.
    """
    try:
        strava_cookie_fetcher = StravaCookieFetcher()
        strava_cookie_fetcher.fetchCookies(email, password)
        cookies = strava_cookie_fetcher.getCookies()
        key_pair_id = cookies['CloudFront-Key-Pair-Id']
        policy = cookies['CloudFront-Policy']
        signature = cookies['CloudFront-Signature']
        return key_pair_id, policy, signature
    except Exception as e:
        raise Exception("ERROR! Retrieving Strava cookies failed! Are your credentials correct?") from e

def update_keys_file(key_pair_id, policy, signature):
    """
    Update the keys.py file with the provided Strava keys.

    Parameters
    ----------
    key_pair_id : str
        The CloudFront key pair ID.
    policy : str
        The CloudFront policy.
    signature : str
        The CloudFront signature.
    """
    with open('keys.py', 'r') as f:
        lines = f.readlines()

    with open('keys.py', 'w') as f:
        for line in lines:
            if line.startswith("KEY_PAIR_ID"):
                f.write(f"KEY_PAIR_ID = '{key_pair_id}'\n")
            elif line.startswith("POLICY"):
                f.write(f"POLICY = '{policy}'\n")
            elif line.startswith("SIGNATURE"):
                f.write(f"SIGNATURE = '{signature}'\n")
            else:
                f.write(line)

def get_strava_cookies():
    """
    Get Strava cookies using credentials from the keys file or user input.

    This function checks if the Strava keys are present in the keys.py file.
    If not, it prompts the user for their Strava credentials, fetches the
    cookies, and updates the keys.py file.

    Returns
    -------
    tuple
        A tuple containing the CloudFront key pair ID, policy, and signature.
    """
    importlib.reload(keys)  # Reload the keys module to get updated values

    try:
        if keys.KEY_PAIR_ID and keys.POLICY and keys.SIGNATURE:
            print("Using existing Strava keys from keys.py")
            return keys.KEY_PAIR_ID, keys.POLICY, keys.SIGNATURE
        else:
            raise AttributeError("Strava keys are not set in keys.py")
    except AttributeError as e:
        print(e)
        print("Please manually set the Strava keys in keys.py or provide your Strava credentials.")

        email = input('Enter your Strava Email Address: ')
        password = getpass('Enter your Strava Password: ')

        try:
            key_pair_id, policy, signature = fetch_strava_cookies(email, password)
            update_keys_file(key_pair_id, policy, signature)
            print("CloudFront-Key-Pair-Id:", key_pair_id)
            print("CloudFront-Policy:", policy)
            print("CloudFront-Signature:", signature)
            return key_pair_id, policy, signature
        except Exception as e:
            print(e)
            return None, None, None
        

# Function to convert latitude/longitude to tile coordinates
def latlon_to_tile(lat, lon, zoom):
    """
    Convert latitude/longitude to tile coordinates.

    Parameters:
    lat (float): Latitude.
    lon (float): Longitude.
    zoom (int): Zoom level.

    Returns:
    tuple: Tile coordinates (xtile, ytile).
    """
    lat_rad = np.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi) / 2.0 * n)
    return (xtile, ytile)


# Function to convert latitude/longitude to pixel coordinates within a tile
def latlon_to_pixel(lat, lon, zoom, xtile, ytile, tile_size=512):
    """
    Convert latitude/longitude to pixel coordinates within a tile.

    Parameters:
    lat (float): Latitude
    lon (float): Longitude
    zoom (int): Zoom level
    xtile (int): X tile coordinate
    ytile (int): Y tile coordinate
    tile_size (int): Tile size in pixels

    Returns:
    tuple: Pixel coordinates (xpixel, ypixel)
    """
    lat_rad = np.radians(lat)
    n = 2.0 ** zoom
    xpixel = int(((lon + 180.0) / 360.0 * n - xtile) * tile_size)
    ypixel = int(((1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi) / 2.0 * n - ytile) * tile_size)
    return (xpixel, ypixel)


# Function to create route coordinates from a path using the detailed geometry
def create_route_coords(G, path):
    """
    Create route coordinates from a path using the detailed geometry.

    Parameters:
    G (networkx.MultiDiGraph): Graph representing the network.
    path (list): List of nodes representing the path.

    Returns:
    list: List of coordinates (latitude, longitude) for the route.
    """
    route_coords = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_data = G.get_edge_data(u, v)
        for key, data in edge_data.items():
            if 'geometry' in data:
                # If the edge has geometry data, extend route_coords with it
                route_coords.extend([(pt[1], pt[0]) for pt in data['geometry'].coords])
            else:
                # If no geometry data, just use the start and end points
                if not route_coords or route_coords[-1] != (G.nodes[u]['y'], G.nodes[u]['x']):
                    route_coords.append((G.nodes[u]['y'], G.nodes[u]['x']))
                route_coords.append((G.nodes[v]['y'], G.nodes[v]['x']))
    return route_coords




# Get the average values of a pixel and its neighbors of level.
def get_neighborhood(matrix, x, y, level=0):
    """
    Get the average values of a pixel and its neighbors of a specified level.

    Parameters:
    matrix (np.ndarray): 2D array representing the image
    x (int): X coordinate of the pixel
    y (int): Y coordinate of the pixel
    level (int): Neighborhood level

    Returns:
    float: Average value of the neighborhood
    """
    neighborhood = []
    for i in range(-level, level + 1):
        for j in range(-level, level + 1):
            if 0 <= x + i < matrix.shape[1] and 0 <= y + j < matrix.shape[0]:
                neighborhood.append(matrix[y + j, x + i])
    return np.mean(neighborhood)


# Example function to calculate the popularity score for a route segment
def calculate_popularity_score(route_coords, tile_image, zoom, xtile, ytile):
    """
    Calculate the popularity score for a route segment.


        As explained in https://medium.com/strava-engineering/the-global-heatmap-now-6x-hotter-23fc01d301de
        the value of a pixel, between 0 and 255, come frem Histogram equalization. 
        That is: it is 255 times the  percentage of pixels with a lower heat value in the 5*5 tiles are around this one. 
        This method yields maximal contrast by ensuring that there are an equal number of pixels of each color. 
        A disadvantage of this approach is that the heatmap is not absolutely quantitative. 
        The same color only locally represents the same level of heat data.

    Parameters:
    route_coords (list): List of route coordinates (lat, lon)
    tile_image (np.ndarray): Tile image as a 2D array
    zoom (int): Zoom level
    xtile (int): X tile coordinate
    ytile (int): Y tile coordinate

    Returns:
    float: Popularity score
    """
    tile_size = tile_image.shape[0]  # pixel number is tile_size * tile_size assuming square tiles
    scores = []
    for coord in route_coords:
        lat, lon = coord
        x, y = latlon_to_pixel(lat, lon, zoom, xtile, ytile, tile_size)
        if 0 <= x < tile_size and 0 <= y < tile_size:  # If out of bounds we do not take into account
            scores.append(get_neighborhood(tile_image, x, y))   # Get the average values of a pixel and its neighbors of level (default 0 so only the piwel is considered)
    # If the scores list is empty, add -1 to it
    if not scores:
        scores = [-1]
    return np.mean(scores)


# Function to add popularity attributes to all edges in the graph.
def add_edge_popularity(G: nx.MultiDiGraph, tile_image, zoom_level, xtile, ytile) -> nx.MultiDiGraph:
    """
    Add popularity attributes to all edges in the graph.

    Parameters:
    G (nx.MultiDiGraph): The input graph.
    tile_image (PIL.Image): Image tile used to calculate popularity.
    zoom_level (int): Zoom level for the tile image.

    Returns:
    nx.MultiDiGraph: The graph with popularity attributes added to all edges.
    """
    u, v, k = zip(*G.edges(keys=True))
    uvk = tuple(zip(u, v, k))

    # Calculate edges' popularity score from u to v
    popularity_scores = []
    for u, v, k in uvk:
        path = [u, v]
        coordinates = create_route_coords(G,path)
        popularity_score = calculate_popularity_score(coordinates, tile_image, zoom_level, xtile, ytile)
        popularity_scores.append(popularity_score)

    # Set the popularity attribute for each edge
    nx.set_edge_attributes(G, dict(zip(uvk, popularity_scores)), name="popularity")

    print("Added popularity attributes to all edges")
    return G


# Function to download a tile with authentication
def download_tile(url, key_pair_id, policy, signature):
    """
    Download a tile with authentication.

    Parameters:
    url (str): URL of the tile.
    key_pair_id (str): Key-Pair-Id for authentication.
    policy (str): Policy for authentication.
    signature (str): Signature for authentication.

    Returns:
    np.array: Tile image as a NumPy array.
    """
    full_url = f"{url}?Key-Pair-Id={key_pair_id}&Policy={policy}&Signature={signature}"
    response = requests.get(full_url)
    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content))
        return np.array(image)
    else:
        raise Exception(f"Failed to download tile: {response.status_code}")



# --------------------------------
# Example usage of the functions
# --------------------------------

def example_fetch_strava_cookies():
    """
    Example function to demonstrate fetching Strava cookies.
    """
    return get_strava_cookies()

if __name__ == "__main__":
    # Run the example function
    example_fetch_strava_cookies()
