# test needed for NWMOA* and NWRCA*

import heapq
import numpy as np


def backward_single_objective_search(G, goal, cost_attr):
    # Reverse the graph
    G_rev = G.reverse()
    
    # Initialize heuristic values
    h = {node: float('inf') for node in G.nodes}
    h[goal] = 0
    
    # Perform single-objective search (Bellman-Ford)
    for _ in range(len(G_rev.nodes) - 1):
        for u, v, data in G_rev.edges(data=True):
            if h[v] + data[cost_attr] < h[u]:
                h[u] = h[v] + data[cost_attr]
    
    return h



def detect_negative_cycles(G, cost_attrs):
    # Check for negative cycles using Bellman-Ford
    for cost_attr in cost_attrs:
        try:
            nx.find_cycle(G, weight=cost_attr)
            return True  # Negative cycle detected
        except nx.NetworkXNoCycle:
            continue
    return False

cost_attrs = ['cost1', 'cost2', 'cost3']
if detect_negative_cycles(G, cost_attrs):
    print("Graph contains negative cycles.")
else:
    print("No negative cycles detected.")



# the heuristic function h_i for the MOSP can be computed by solving the shortest path problem for each cost dimension from the goal node, with the graph edges reversed. 
# This provides a lower bound on the cost from any node to the goal, serving as a perfect heuristic for the A* search algorithm.
    
def compute_heuristic(graph, goal, num_objs):
    """
    Computes the heuristic function for each cost dimension for MOSP.
    
    Parameters:
    graph (dict): The input graph where each node points to a list of tuples (neighbor, costs).
    goal (int): The goal node.
    num_objs (int): The number of cost dimensions.
    
    Returns:
    np.ndarray: A 2D array where heuristic[i][j] is the heuristic value for node i on cost dimension j.
    """
    num_nodes = len(graph)
    heuristic = np.full((num_nodes, num_objs), np.inf)
    
    for obj_index in range(num_objs):
        # Initialize the priority queue and distances
        pq = [(0, goal)]
        heuristic[goal][obj_index] = 0
        
        while pq:
            curr_cost, node = heapq.heappop(pq)
            
            if curr_cost > heuristic[node][obj_index]:
                continue
            
            for neighbor, costs in graph[node]:
                cost = costs[obj_index]
                new_cost = curr_cost + cost
                
                if new_cost < heuristic[neighbor][obj_index]:
                    heuristic[neighbor][obj_index] = new_cost
                    heapq.heappush(pq, (new_cost, neighbor))
    
    return heuristic

'''  
# Example usage
graph = {
    0: [(1, [1, 2]), (2, [4, 1])],
    1: [(2, [2, 3]), (3, [5, 2])],
    2: [(3, [1, 1])],
    3: []
}

goal = 3
num_objs = 2

heuristic = compute_heuristic(graph, goal, num_objs)
print("Heuristic values:")
print(heuristic)

'''
