# =============================================================================
# Algorithms
# Mainly Constrained Shortest Path (CSP) or Bi/Multi-Objective Search Algorithm
# =============================================================================

'''
Modified Dijkstra's Algorithm, Yen's algorithm or Bellman of Johnson or 
A* Search Algorithm,
Dyamical Programing (dp[u][d] can represent the maximum elevation gain to reach node u with a distance d),
Mixed-Integer Linear Programming (MILP) i=with CPLEX or Gurobi

Breadth-First Search (BFS)
Depth-First Search (DFS)
Constrained Shortest Path First (CSPF)  
Lagrangian Relaxation Techniques


More formally, in multi-objective search problem (MOSP) or multiobjective path problem, see also Resource Constrained Shortest Path Problem (RCSPP) or 
shortest path problem with resource constraints (SPPRC) (a well known NP-hard problem), we are given a
directed graph with multiple costs annotating each edge, a
specified start state, and a specified goal state. A path π is
considered to be better  than, i.e., to dominate, another path π′
if and only if π is not worse than π′ on any cost metric and π is better than π′ 
on at least one cost metric, and a Pareto optimal solution (also called efficient) is a path from the start state to the goal state
that are not dominated by any path from the start state to the
goal state. 
We can ompute the set of all Pareto-optimal solutions (one good algorithm for bi-objective search being BOA*) or only a subset of
efficient paths that is good enough. This motivates the study of Fully
Polynomial Time Approximation Schemes (FPTAS) for MOSP problem.
A good solution is to find the so called minimal complete set of efficient paths (One-to-One Multiobjective Shortest Path Problem):
 that is to find a representative efficient path for every attribute (non-dominated cost vector). The so called NAMOA∗dr (or NAMOA∗dr-lazy) algorithm is the state of the art One-to-One MOSP algorithm in the literature

Some article in the litterature are given in the Repository (see also the review: 
"Review Multiobjective Path Problems and Algorithms in Telecommunication Network Design—Overview and Trends")

The best one are: 2024 Exact Multi-objective Path Finding with NegativeWeights 
and may be the others. Check what kind of Pareto set it calculates (total, minimal, approximate = espilon Pareto set, ...)

Some repositories with codes exists
cf: https://github.com/torressa/cspy (A collection of algorithms for the (resource) Constrained Shortest Path (CSP) problem)
https://github.com/pathwyse
https://zenodo.org/records/7702018 (Targeted Multiobjective Dijkstra Algorithm)


Some are specific for bike problem like the heuristic-enabled Dijkstra algorithm developed by Hrncir et al.
(2017) or Evolutionary algorithm. A good review is given by 2023 Solving the multi-objective bike routing problem by meta heuristic algorithm Nunes Sanstos IntlTransOpRes

'''



import networkx as nx
from collections import deque
import heapq

def bfs_min_distance_with_min_elevation(graph, start, goal, min_elevation_gain):
    # Queue: (current node, total distance, elevation gain, path)
    queue = deque([(start, 0, 0, [start])])
    visited = set()

    while queue:
        node, total_distance, elevation_gain, path = queue.popleft()

        if node in visited:
            continue
        visited.add(node)

        if node == goal and elevation_gain >= min_elevation_gain:
            return path, total_distance, elevation_gain

        for neighbor in graph.neighbors(node):
            edge_data = graph.get_edge_data(node, neighbor,0)
            if edge_data == 0:
                continue  # Skip to the next neighbor
            edge_distance = float(edge_data['length'])
            edge_elevation = float(edge_data['height_gain'])

            new_total_distance = total_distance + edge_distance
            new_elevation_gain = elevation_gain + edge_elevation

            # Enqueue the new state if it meets the elevation gain constraint
            if new_elevation_gain >= min_elevation_gain:
                queue.append((neighbor, new_total_distance, new_elevation_gain, path + [neighbor]))

    return None, float('inf'), 0



def a_star_max_elevation(graph, start, goal, max_distance, elevation):
    # Priority queue: (negative elevation gain, current cost, current node, path)
    pq = [(-elevation[start], 0, start, [start])]
    visited = set()

    while pq:
        neg_gain, cost, node, path = heapq.heappop(pq)
        gain = -neg_gain

        if node in visited:
            continue
        visited.add(node)

        if node == goal:
            return path, gain

        for neighbor in graph.neighbors(node):
            if neighbor in visited:
                continue

            edge_data = graph.get_edge_data(node, neighbor,0)
            if edge_data == 0:
                continue  # Skip to the next neighbor

            edge_distance = edge_data['length']
            edge_elevation = edge_data['height_gain']

            new_cost = cost + edge_distance
            new_gain = gain + edge_elevation

            if new_cost <= max_distance:
                heapq.heappush(pq, (-new_gain, new_cost, neighbor, path + [neighbor]))

    return None, 0


def dijkstra_with_elevation_constraint(graph, start, goal, min_elevation_gain):
    # Priority queue: (distance, current node, elevation gain, path)
    pq = [(0, start, 0, [start])]
    visited = set()
    min_distance = {start: 0}
    max_elevation_gain = {start: 0}

    while pq:
        dist, node, gain, path = heapq.heappop(pq)

        if node in visited:
            continue
        visited.add(node)

        if node == goal and gain >= min_elevation_gain:
            return path, dist, gain

        for neighbor in graph.neighbors(node):
            edge_data = graph.get_edge_data(node, neighbor,0)
            if edge_data == 0:
                continue  # Skip to the next neighbor

            edge_distance = float(edge_data['length'])
            edge_elevation = float(edge_data['height_gain'])

            new_dist = dist + edge_distance
            new_gain = gain + edge_elevation

            if new_dist < min_distance.get(neighbor, float('inf')) or new_gain > max_elevation_gain.get(neighbor, float('-inf')):
                min_distance[neighbor] = new_dist
                max_elevation_gain[neighbor] = new_gain
                heapq.heappush(pq, (new_dist, neighbor, new_gain, path + [neighbor]))

    return None, float('inf'), 0


def path_elevation_gain(G, path, elevation_attribute='height_gain'):
    """
    Calculate the total elevation gain of a path.
    """
    total_elevation = 0
    for i in range(len(path) - 1):
        edge_data = G.get_edge_data(path[i], path[i + 1])
        total_elevation += edge_data.get(elevation_attribute, 0)
    return total_elevation

def shortest_path_with_min_elevation_brute_force(G, source, target, min_elevation_gain, weight=None, elevation_attribute='height_gain'):
    """
    Find the shortest path from source to target with at least the minimum elevation gain.
    """
    for path in nx.shortest_simple_paths(G, source, target, weight=weight):
        elevation_gain = path_elevation_gain(G, path, elevation_attribute)
        if elevation_gain >= min_elevation_gain:
            return path, elevation_gain
    return None, None


