# =============================================================================
# Algorithms
# Mainly Constrained Shortest Path (CSP) or Bi/Multi-Objective Search Algorithm
# =============================================================================

'''
We have started form these algorithm:
Modified Dijkstra's Algorithm, Yen's algorithm or Bellman of Johnson or Floyd–Warshall
A* Search Algorithm,
Dynamic Programming (DP) (dp[u][d] can represent the maximum elevation gain to reach node u with a distance d),
Mixed-Integer Linear Programming (MILP) with CPLEX or Gurobi
Breadth-First Search (BFS)
Depth-First Search (DFS)
Constrained Shortest Path First (CSPF)  
Lagrangian Relaxation Techniques


But the problem is in fact more general cf https://en.wikipedia.org/wiki/Constrained_optimization or better
https://en.wikipedia.org/wiki/Multi-objective_optimization

More formally we are in the or multi-objective optimization (MOO) area.
A lot of work focus on bi-objectif, or multi (meaning 2,3)-objective optimization techniques we stay general
with  multi-objective search problem (MOSP) (that is with 4,5 .. more objectif), or multiobjective path problem or Constrained multi-objective optimization problems (CMOPs), 
see also Resource Constrained Shortest Path Problem (RCSPP) or 
shortest path problem with resource constraints (SPPRC) (a well known NP-hard problem), we are given a
directed graph with multiple costs annotating each edge, a
specified start state, and a specified goal state. 
We also have: 
Shortest Path Problems with Time Windows (SPPTWs)
Resource-Constrained Shortest Path Problems (RCSPPs)
Resource-Constrained Elementary Shortest Path Problems (RCESPPs)
Shortest Path Problems with Forbidden Paths (SPPFPs)


A path π is
considered to be better  than, i.e., to dominate, another path π′
if and only if π is not worse than π′ on any cost metric and π is better than π′ 
on at least one cost metric, and a Pareto optimal solution (also called efficient) is a path from the start state to the goal state
that are not dominated by any path from the start state to the
goal state. 
Sometimes we can add constrains (max, min of some cost function) so The multiobjective path problem can be formulated as an optimization program with linear constraints.
Another similar problem is bi-objective shortest path problem (BOSPP) or the weight constrained shortest path problem (WCSPP): find a minimum-cost (shortest) path between two points such that the total weight (or resource consumption) of the path is limited.

We can compute the set of all Pareto-optimal solutions (one good algorithm for bi-objective search being BOA*) 
or only the so called minimal complete set of efficient paths (sometimes called One-to-One Multiobjective Shortest Path Problem):
 that is to find a representative efficient path for every attribute (non-dominated cost vector). 
 New Approach to Multi-Objective A*:   NAMOA∗dr (or NAMOA∗dr-lazy) algorithm is the state of the art One-to-One MOSP algorithm in the literature
These algorithms are when the  objective function is additive (sum of the cost value per edge) but some are more general (ex cost = cost_path1/cost path_2) cf 2024 Multiobjective (for efficient algorithm)

An important consideration is that the lists of sored path, nodes, ... needed to perfome the calcul can grow exponentially

Finally another way is to find only a subset of efficient paths that is good enough. This motivates the study of 
* Fully Polynomial Time Approximation Schemes (FPTAS) for MOSP problem.
* Heuristic and Metaheuristic Approaches: like simulated annealing (SA) and tabu search (TS) 
Evolutionary Algorithms (EA):   Multi-Objective Artificial Bee Colony (MOABC) and Non-Dominant Sorting Genetic Algorithm II (NSGA-II) 
(but  a LOT of study cf multi-strategy adaptable ant colony optimization (MsAACO) or prominent swarm optimization (PSO))
SO now the NSGA-III (that is NSGA-II for multi objective, see also  Inexpensive Constraint Surrogate-assisted Non-dominated Sorting Genetic
Algorithm (IC-SA-NSGA-II) or R-NSGA-III, or in Bayesian approach Self-Adaptive Algorithm for Multi-Objective Constraint Optimization by using Radial Basis Function Approximations (SAMO-COBRA)
MOEA/D or MOEAD (Multiobjective Evolutionary Algorithm Based on Decomposition) + variant (such as ε-MOEA (ε-Domination Based Multi-Objective Evolutionary Algorithm)) and 
Strength Pareto Evolutionary Algorithm SPEA2-SDE  are the state of the Art models (cf Wikipedia or 2024 Springer Review). But 
Multi-Objective Particle Swarm Optimization (MOPSO) and Differential Evolution (DE) are alos Popular due to their simplicity.
* AI Techniques: Methods like Deep Neural Networks (DNN) and Fuzzy Inference Systems (FIS) 
* GRASP: The Greedy Randomized Adaptive Search Procedure 


We have to distinguish between the following algotithm also:
* No-preference methods: Neutral solution found without DM (Descision Making by Human). A similar idea is folowed by lexicographic ordering to incorporate priorities of the
objectives in order of importance. 
Also are the Pruning methods to reduce the number of Pareto optimal solutions using predefined rules (diversity, not too dense, hypervolume, ...).
* A priori methods: DM gives preferences first, solution found to match.
* A posteriori methods: Pareto solutions provided, DM selects preferred one.
* Interactive methods: DM iteratively refines solutions with feedback.
and many  others depending on the kind of Pareto set it calculates (total, minimal, approximate = espilon Pareto set, ...).
Without talking about Route Planning Algorithms (include traffic for instance) https://wiki.openstreetmap.org/wiki/Routing or noise or Networking etc... with the popular  ORSM, Graphhopper,  BRouter, valhalla, 




Some article in the litterature are given in the Repository (see also the review: 
"Review Multiobjective Path Problems and Algorithms in Telecommunication Network Design—Overview and Trends")

for instance A-A*pex is fond similar to LTMOA*  but this one was overpass by NWMOA*

The best articles (especiallly for reference therin) are: 
2024 SkiVis: Visual Exploration and Route Planning in Ski Resorts (good revew for sky, hiking, biking)
2024 Evolutionary constrained multi‑objective optimization Review Liang Yu Vicinagerath
2024 Exact Multi-objective Path Finding with NegativeWeights (for some new algorythms): 
2024 Multiobjective Path Problems and Algorithms in Telecommunication Network Design—Overview and Trends
2024 Constrained multi-objective optimization problems .. (unfortunatly does not realy compare the performances)
2024 A review of Pareto pruning methods for multi-objective optimization (good for the methods, but does not compare algorithm)
2023 Enhanced methods for the weight constrained shortest path problem (compare BiPulse,WC-EBBA*, WC-BA*, RC-BDA*, Pulse and CSP)
2023 Heuristic-Search Approaches for the Multi-Objective Shortest-Path Problem: Progress and Research Opportunities  (very good review)
2023 Solving the multi-objective bike routing problem by meta heuristic algorithm (propsed the NGA-MOBRP: new genetic approach for Multi-Objective Bike Routing Problem)
2022 A review and evaluation of multi and many-objective optimization: Methods and algorithms 
2022 Application of state‑of‑the‑art multiobjective metaheuristic algorithms in reliability‑based design optimization: a comparative study (good reveiw but I did not download it because no best algorithm is found)
+ older 2011: Multiobjective evolutionary algorithms: A survey of the state of the art; 2015: Many-objective evolutionary algorithms: A survey  (veyr god to see who cite them ..)
and a very good one is the specific for bike problem: heuristic-enabled Dijkstra algorithm developed by Hrncir et al. 
and may article on cycle-tourism like "2023 Sustainable cycle-tourism for society: Integrating multi-criteria decision-making and land use approaches for route selection"
or "2022 Multi-objective Route Planning Problem for Cycle-tourists""

adn the pioneerd one "Algorithms for finding paths with multiple constraints"



In python: tons of codes (not laking about the general optmization:  SciPy.optimize  pyOpt, Pyomo): cspy, PyGMO (much better is https://esa.github.io/pygmo2/), pyMCMA, GPOL, pyMultiobjective ( a very good one with all references), paretoset, pathwyse (in C++)... BUt the best one seems to be:
* DEAP (Distributed Evolutionary Algorithms in Python) with Multi-objective optimisation (NSGA-II, NSGA-III, SPEA2, MO-CMA-ES): https://github.com/DEAP/deap or https://pypi.org/project/deap/
* Platypus NSGA-II, NSGA-III, MOEA/D, IBEA, Epsilon-MOEA, SPEA2, GDE3, OMOPSO, SMPSO, and Epsilon-NSGA-II: https://github.com/Project-Platypus/Platypus  https://pypi.org/project/Platypus-Opt/    https://platypus.readthedocs.io/en/latest/
* pymoo: Multi-objective Optimization in Python  : GA, DE, BRKGA, NelderMead, PatternSearch, CMAES, ES, SRES, ISRES, NSGA-II, R-NSGA-II, NSGA-III, U-NSGA-III, R-NSGA-III, MOEAD, AGE-MOEA, C-TAEA, SMS-EMOA, and RVEA.  https://pymoo.org/
+ interesting to test
https://zenodo.org/records/7702018 (Targeted Multiobjective Dijkstra Algorithm + NAMOA_lazy + ..) in C++
https://bitbucket.org/s-ahmadi (for insatnce with the NWMOA* that seems to be the best algorithm for the Exact Multi-objective Path Finding)



So in summary:  be careful  to choose the best lexicographic order for the cost: this has impact (cf 2023 Heuristic) 
* NGA-MOBRP: (2023 article) is the most suitable to be employed within a real time tool for cyclists: good quality metrics in a reasonable computational time. Slightly faster than Multi-Objective
Simulated Annealing (MOSA) approach
* NWMOA* (2024 Exact..) an exact MOSP algorithm. Way beter than T-MDA: Targeted Multiobejctive Dijkstra algorithm, that was similar to NAMOA∗ dr-lazy to find minimal pareto set
* ERCA*: (2023 A New Approach for the Resource Constrained Shortest Path Problem): Enhanced Resource Constrained A* (ERCA*): way faster than BiPulse, an existing leading algorithm for RCSPP cf https://github.com/rap-lab-org/public_erca

* WC-EBBA*par (enhanced biased bidirectional A*: paralelism for multi core) (2024 enhanced) is a great choice for applications that need fast solution approaches;
    it is based on RC-BDA* and BiPulse and is bidirectioal,  contrary to the single directional: WC-A*  cf https://bitbucket.org/s-ahmadi/biobj/src/master/

See also the above one such as multi-objective evolutionary algorithms (MOEAs) and
*  the ε constrained method and Adaptive operator selection (AOS) are used in Multiobjective evolutionary algorithm based on decomposition (MOEA/D) (2014)
indeed: ε-MOEA has been successful in finding well-converged and well-distributed solutions with a much smaller computational effort 
than a number of state-of-the-art MOEAs including NSGA-II, SPEA2, and PESA (as quoted by Kalyanmoy Deb)

'''



import networkx as nxn
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


