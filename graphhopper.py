import requests

# Set up the API endpoint and parameters
url = 'https://graphhopper.com/api/1/route'
params = {
    'point': ['52.5160,13.3779', '52.5206,13.3862'],  # Start and end points
    'vehicle': 'bike',
    'locale': 'en',
    'key': 'd59cedaa-f233-4435-b486-73cc3674f158'  # Your GraphHopper API key
}

# Make the request
response = requests.get(url, params=params)

# Process the response
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Error: {response.status_code}")





def custom_meta_heuristic(route_data):
    # Implement your algorithm here
    optimized_route = route_data  # Placeholder for the optimized route
    return optimized_route

# Use GraphHopper API to get the initial route
response = requests.get(url, params=params)
if response.status_code == 200:
    data = response.json()
    optimized_route = custom_meta_heuristic(data)
    print(optimized_route)
else:
    print(f"Error: {response.status_code}")
    
    
    
    
    
    
import requests
import folium

# Set up the API endpoint and parameters
url = 'https://graphhopper.com/api/1/route'
params = {
    'point': ['52.5160,13.3779', '52.5206,13.3862'],  # Start and end points
    'vehicle': 'bike',
    'locale': 'en',
    'key': 'd59cedaa-f233-4435-b486-73cc3674f158',  # Your GraphHopper API key
    'points_encoded': 'false'  # Get the points in a readable format
}

# Make the request
response = requests.get(url, params=params)
if response.status_code == 200:
    data = response.json()
    # Extract the points
    points = data['paths'][0]['points']['coordinates']
else:
    print(f"Error: {response.status_code}")
    points = []

# Create a folium map centered at the start point
start_coords = [52.5160, 13.3779]
m = folium.Map(location=start_coords, zoom_start=14)

# Add the route to the map
folium.PolyLine(points, color='blue', weight=5, opacity=0.7).add_to(m)

# Save the map to an HTML file
m.save('route_map.html')