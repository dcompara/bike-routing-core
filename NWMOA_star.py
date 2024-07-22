########################################
# unidirectional A* search
########################################

# cf https://bitbucket.org/s-ahmadi/multiobj/src/main/
# From Exact Multi-objective Path Finding with Negative Weights: https://doi.org/10.1609/icaps.v34i1.31455



import networkx as nx
import heapq


# the heuristic function h_i for the MOSP can be computed by solving the shortest path problem for each cost dimension from the goal node, with the graph edges reversed. 
# This provides a lower bound on the cost from any node to the goal, serving as a perfect heuristic for the A* search algorithm.
def compute_heuristics(G, goal, num_objs, weights):
    heuristics = {i: {} for i in range(num_objs)}
    for i in range(num_objs):
        for node in G.nodes:
            heuristics[i][node] = nx.single_source_bellman_ford_path_length(G.reverse(), goal, weight=weights[i])
    return heuristics


# Check if vector a is lexicographically less than or equal to vector b.
def lexicographical_less_equal(a, b):
    """
    Check if vector a is lexicographically less than or equal to vector b.

    Args:
        a (list): The first vector.
        b (list): The second vector.

    Returns:
        bool: True if a is lexicographically less than or equal to b, False otherwise.
    """
    for i in range(len(a)):
        if a[i] < b[i]:
            return True
        elif a[i] > b[i]:
            return False
    return True



 
def is_dominated(v, V):
    """
    Check if a given cost vector (v) is weakly dominated by any vector in a list of cost vectors (V).

    Args:
        v (list): The cost vector to check for dominance.
        V (list of lists): A list of cost vectors to compare against.

    Returns:
        bool: True if the vector v is weakly dominated by any vector in V, False otherwise.
    """
    for v_prime in V:
        # Check if v_prime is not lexicographically less than v
        if not lexicographical_less_equal(v_prime, v):
            return False
        
        # Check if v_prime is weakly dominated by v
        if all(v_prime[i] <= v[i] for i in range(len(v))):
            return True

    # If no cost vector in V weakly dominates v, return False
    return False









###########################################################################
# # NWMOA* : A unidirectional A* search for the MOSPP and negative weights
###########################################################################



'''
Overview of NWMOA* Algorithm (by GPT4o)

Overview of NWMOA* Algorithm
NWMOA* is a unidirectional multi-objective A* search algorithm designed to solve the Multi-Objective Shortest Path Problem (MOSP) with multiple cost attributes, including negative weights and negative cycles. The goal is to find all Pareto-optimal paths from a start node to a goal node in a graph with multiple cost dimensions.

Steps Involved in NWMOA*
Initialization:

Graph Representation: The graph G and its reverse G_rev are provided as inputs.
Parameters: The number of objectives (cost attributes), number of vertices, and various arrays for storing heuristic values, truncated cost vectors, and labels are initialized.
Label Pool: A pool of labels is created to manage the labels used during the search.

Preliminary Calculations:
Breadth-First Search (BFS): A BFS is performed from the start node to identify reachable vertices and their distances.
Negative Cycle Detection: The algorithm uses a backward Dijkstra's algorithm (or Bellman-Ford) to calculate lower bounds and check for negative cycles in the graph. If a negative cycle is detected, the algorithm terminates early.
Max Delta Calculation: The algorithm calculates the largest Delta f-value for nodes in the priority queue during any iteration of A*.

Main Multi-Objective A Search*:
Priority Queue Initialization: A priority queue (using a cyclic bucket queue) is initialized to manage the nodes to be expanded during the search.

Label Generation and Expansion:
The algorithm generates the first label for the start node and inserts it into the priority queue.
While the priority queue is not empty, the algorithm extracts the node with the least cost.
For the extracted node, it performs quick dominance checks and full dominance checks to prune dominated labels.
Non-dominated labels are stored and the node is expanded by generating new labels for its successors.
Solution Capture: When the goal node is reached, the path cost is stored and any dominated solutions are pruned.

Result Storage and Printing:
After the search is complete, the algorithm calculates the memory used, stores the results, and prints statistics about the search and the solutions found.

Key Concepts:
Dominance Check: This is used to prune labels that are dominated by others (i.e., labels that do not offer any cost improvement in any dimension).
Priority Queue: Manages the order in which nodes are expanded based on their costs. The queue ensures that nodes with lower costs are expanded first.
Label Pool: Recycles labels to manage memory efficiently during the search.
Truncated Labels: Truncated cost vectors are used for efficient dominance checks and to reduce the dimensionality of the search space.

'''


def nwmoa_star(G, start, goal, num_objs, weights):
    heuristics = compute_heuristics(G, goal, num_objs, weights) # directly computes the heuristics without an explicit BFS step: implicitly handles initialization by computing heuristics for all node
    open_list = []
    heapq.heappush(open_list, (0, start, [0]*num_objs, []))
    closed_list = {}
    solutions = []

    while open_list:
        f, current, g, path = heapq.heappop(open_list)
        path = path + [current]
        
        if current == goal:
            solutions.append((g, path))
            continue
        
        if current in closed_list and is_dominated(g, closed_list[current]):
            continue
        
        if current not in closed_list:
            closed_list[current] = []
        closed_list[current].append(g)

        for neighbor in G.neighbors(current):
            edge_data = G.get_edge_data(current, neighbor)
            g_new = [g[i] + edge_data[weights[i]] for i in range(num_objs)]
            f_new = sum(g_new[i] + heuristics[i][neighbor] for i in range(num_objs))
            heapq.heappush(open_list, (f_new, neighbor, g_new, path))

    return solutions


'''

# Example usage
G = nx.DiGraph()
# Add nodes and edges to the graph G
# G.add_edge(u, v, weight=w)

start = 'A'  # Starting node
goal = 'B'  # Goal node
num_objs = 2  # Number of objectives

solutions = nwmoa_star(G, start, goal, num_objs)
for solution in solutions:
    print(f"Cost: {solution[0]}, Path: {solution[1]}")
    
'''







###########################################################################
#  NWRCA* : A unidirectional A* search for the RCSPP and negative weights 
###########################################################################


'''
Overview of NWRCA* Algorithm (by GPT4o)

NWRCA* is a unidirectional A* search algorithm designed to solve the Resource Constrained Shortest Path Problem (RCSPP) with multiple cost attributes and negative weights. 
The goal is to find the shortest path from a start node to a goal node while considering multiple cost attributes and ensuring no negative cycles are present.

Steps Involved in NWRCA*

Initialization:
Graph Representation: The graph G and its reverse G_rev are provided as inputs.
Parameters: The number of objectives (cost attributes), number of vertices, and various arrays for storing heuristic values, upper bounds, and budget constraints are initialized.
Label Pool: A pool of labels is created to manage the labels used during the search.

Preliminary Calculations:
Breadth-First Search (BFS): A BFS is performed from the start node to identify reachable vertices and their distances.
Negative Cycle Detection: The algorithm uses a backward Dijkstra's algorithm (or Bellman-Ford) to calculate lower bounds and check for negative cycles in the graph. If a negative cycle is detected, the algorithm terminates early.

Budget Setup:
Based on a given constraint, the algorithm sets up resource budgets for non-primary costs. This ensures that the search remains within specified resource limits.

Main Multi-Objective A Search*:
Priority Queue Initialization: A priority queue (using a cyclic bucket queue) is initialized to manage the nodes to be expanded during the search.

Label Generation and Expansion:
The algorithm generates the first label for the start node and inserts it into the priority queue.
While the priority queue is not empty, the algorithm extracts the node with the least cost.
For the extracted node, it performs quick dominance checks and full dominance checks to prune dominated labels.
Non-dominated labels are stored and the node is expanded by generating new labels for its successors.
Solution Capture: When the goal node is reached, the path cost is stored and any dominated solutions are pruned.

Result Storage and Printing:
After the search is complete, the algorithm calculates the memory used, stores the results, and prints statistics about the search and the solutions found.

Key Concepts:
Dominance Check: This is used to prune labels that are dominated by others (i.e., labels that do not offer any cost improvement in any dimension).
Priority Queue: Manages the order in which nodes are expanded based on their costs. The queue ensures that nodes with lower costs are expanded first.
Label Pool: Recycles labels to manage memory efficiently during the search.
Budget Constraints: Ensures that the search stays within the specified resource limits for non-primary costs.

'''

# NEED TO BE CHECKED 

def nwrca_star(G, start, goal, num_objs, constraints, weight='weight'):
    heuristics = compute_heuristics(G, goal, weight)
    open_list = []
    heapq.heappush(open_list, (0, start, [0]*num_objs, []))
    closed_list = {}
    solutions = []
    budgets = [float('inf')] * num_objs

    while open_list:
        f, current, g, path = heapq.heappop(open_list)
        path = path + [current]
        
        if current == goal:
            solutions.append((g, path))
            budgets[0] = min(budgets[0], g[0])
            continue
        
        if current in closed_list and is_dominated(g, closed_list[current]):
            continue
        
        if current not in closed_list:
            closed_list[current] = []
        closed_list[current].append(g)

        for neighbor in G.neighbors(current):
            edge_data = G.get_edge_data(current, neighbor)
            g_new = [g[i] + edge_data.get(weight, 1) for i in range(num_objs)]
            if g_new[0] > budgets[0] or any(g_new[i] > constraints[i] for i in range(1, num_objs)):
                continue
            f_new = sum(g_new[i] + heuristics[neighbor][i] for i in range(num_objs))
            heapq.heappush(open_list, (f_new, neighbor, g_new, path))

    return solutions



''' 

# Example usage
G = nx.DiGraph()
# Add nodes and edges to the graph G
# G.add_edge(u, v, weight=w)

start = 'A'  # Starting node
goal = 'B'  # Goal node
num_objs = 2  # Number of objectives
constraints = [float('inf'), 100]  # Example resource constraints

solutions = nwrca_star(G, start, goal, num_objs, constraints)
for solution in solutions:
    print(f"Cost: {solution[0]}, Path: {solution[1]}")


'''


