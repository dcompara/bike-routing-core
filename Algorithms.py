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

But if we use a single objective funciton using linearization of many objective using weights. This always result in solutions distributed a small area on the entire Pareto front (Boyd & Vandenberghe, 2004).
This is also called scalarization with Equal, Rank Sum, Rank Exponent, centroid, inverse weight method, or analytic hierarchy process method (AHP) ...: cf 2013 ANK ORDERING CRITERIA WEIGHTING METHODS
– A COMPARATIVE OVERVIEW) or "Multi-Criteria Decision Making (MCDM) Methods and Concepts"


But the problem is in fact more general cf https://en.wikipedia.org/wiki/Constrained_optimization or better
https://en.wikipedia.org/wiki/Multi-objective_optimization
cf book by "Multiple attribute decision making: methods and applications"
or even the review "2023: Multi-Criteria Decision Making (MCDM) Methods and Concepts"

More formally we are in the or multi-objective optimization (MOO) area.
A lot of work focus on bi-objectif, or multi (meaning 2,3)-objective optimization techniques we stay general
with  multi-objective search problem (MOSP) (that is with 4,5 .. more objectif), or
 multiobjective path problem or Constrained multi-objective optimization problems (CMOPs), 
see also Resource Constrained Shortest Path Problem (RCSPP) or 
shortest path problem with resource constraints (SPPRC) (a well known NP-hard problem), we are given a
directed graph with multiple costs annotating each edge, a
specified start state, and a specified goal state. 
We also have: 
Shortest Path Problems with Time Windows (SPPTWs)
Resource-Constrained Shortest Path Problems (RCSPPs)
Resource-Constrained Elementary Shortest Path Problems (RCESPPs)
Shortest Path Problems with Forbidden Paths (SPPFPs)

Another name for all this is Multi-Constraint Shortest Path (MCSP). See all work by Xiaofang Zhou's team and Ziyi Liu's thesis (a very good start)
they do nt use the name pareto set but "skyline path problem"

This is also stronly related to the "knapsack problem" (cf fully polynomial-time approximation scheme (FPTAS) algorithm given in  "Approximating single- and multi-objective nonlinear sum and
product knapsack problems") or to the Orienteering Problem (OP): a routing problem where the goal is to determine a subset of nodes to visit and in which order, so that the total collected score is maximized without exceeding a given time budget. 
in fact our case is the Arc Orienteering Problem (AOP). With here again  Exact, Heuristic, Metaheuristic or Hybrid Approaches
cf 2016 review http://dx.doi.org/10.1016/j.ejor.2016.04.05 (cited 712)

A path π is
considered to be better  than, i.e., to dominate, another path π′
if and only if π is not worse than π′ on any cost metric and π is better than π′ 
on at least one cost metric, and a Pareto optimal solution (also called efficient) is a path from the start state to the goal state
that are not dominated by any path from the start state to the
goal state. 
Sometimes we can add constrains (max, min of some cost function) so The multiobjective path problem can be formulated
 as an optimization program with linear constraints.
Another similar problem is bi-objective shortest path problem (BOSPP) or the weight constrained shortest path problem (WCSPP): 
find a minimum-cost (shortest) path between two points such that the total weight (or resource consumption) of the path is limited.

We can compute the set of all Pareto-optimal solutions (one good algorithm for bi-objective search being BOA*) 
or only the so called minimal complete set of efficient paths (sometimes called One-to-One Multiobjective Shortest Path Problem):
 that is to find a representative efficient path for every attribute (non-dominated cost vector). 
 New Approach to Multi-Objective A*:   NAMOA∗dr (or NAMOA∗dr-lazy) algorithm is the state of the art One-to-One MOSP algorithm in the literature
These algorithms are when the  objective function is additive (sum of the cost value per edge) but 
some are more general (ex cost = cost_path1/cost path_2) cf 2024 Multiobjective (for efficient algorithm). 

An important consideration is that the lists of sored path, nodes, ... needed to perfome the calcul can grow exponentially

Finally another way is to find only a subset of efficient paths that is good enough. This motivates the study of 
* Fully Polynomial Time Approximation Schemes (FPTAS) for MOSP problem.
* Heuristic and Metaheuristic Approaches: like simulated annealing (SA) and tabu search (TS) 
Evolutionary Algorithms (EA):   Multi-Objective Artificial Bee Colony (MOABC) and Non-Dominant Sorting Genetic Algorithm II (NSGA-II) 
(but  a LOT of study cf multi-strategy adaptable ant colony optimization (a multi-strategy adaptable ant colony optimization (MsAACO: https://doi.org/10.1016/j.knosys.2024.111459)
 ,  see https://medium.com/ai4sm/personalized-cycling-path-routing-cc3c484da2a6) or prominent swarm optimization (PSO))
SO now the NSGA-III (that is NSGA-II for multi objective, see also  Inexpensive Constraint Surrogate-assisted Non-dominated Sorting Genetic
Algorithm (IC-SA-NSGA-II) or R-NSGA-III, or in Bayesian approach Self-Adaptive Algorithm for Multi-Objective Constraint Optimization by using Radial Basis Function Approximations (SAMO-COBRA)
MOEA/D or MOEAD (Multiobjective Evolutionary Algorithm Based on Decomposition) + variant (such as ε-MOEA (ε-Domination Based Multi-Objective Evolutionary Algorithm)) and 
Strength Pareto Evolutionary Algorithm SPEA2-SDE  are the state of the Art models (cf Wikipedia or 2024 Springer Review). But 
Multi-Objective Particle Swarm Optimization (MOPSO) and Differential Evolution (DE) are alos Popular due to their simplicity.
* AI Techniques: Methods like Deep Neural Networks (DNN) and Fuzzy Inference Systems (FIS) 
* GRASP: The Greedy Randomized Adaptive Search Procedure (as used in the 2017 article "Bicycle network design: model and solution algorithm"  
or together with an imporved Iterated Local Search (ILS) in "An arc orienteering algorithm to find the most scenic path
on a large-scale road network") or the orieted arc cycle trip planning problem (CTPP) 2014 10.1016/j.tre.2014.05.006 (wich as we want "upper and a lower bound on the total length." but is still a single (global) objectives)

And obviously we no not look for "non-shortest diverse routes" or "Global Routing Optimization problem that aims to minimize traffic congestion" (so we do not deal with groups etc...)


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
and the "2022 A systematic literature review for the tourist trip design problem: Extensions, solution techniques and future research lines "

for instance A-A*pex is fond similar to LTMOA*  but this one was overpass by NWMOA*   

The best articles (especiallly for reference therin) are: + adding the 2023 thesis A Study of Multi-Constraint Shortest Path Queries in Road Network
2024 SkiVis: Visual Exploration and Route Planning in Ski Resorts (good revew for sky, hiking, biking: that are often  lke "Urban Bike Route Planner" see also "https://doi.org/10.1016/j.inffus.2024.102413")
2024 Evolutionary constrained multi‑objective optimization Review Liang Yu Vicinagerath
2024 Exact Multi-objective Path Finding with NegativeWeights (for some new algorythms): 
2024 Multiobjective Path Problems and Algorithms in Telecommunication Network Design—Overview and Trends (good review and give good "generic algorithms" which can be very nice to read again)
2024 Constrained multi-objective optimization problems .. (unfortunatly does not realy compare the performances)
2024 A review of Pareto pruning methods for multi-objective optimization (good for the methods, but does not compare algorithm)
2023 Enhanced methods for the weight constrained shortest path problem (compare BiPulse,WC-EBBA*, WC-BA*, RC-BDA*, Pulse and CSP)
2023 Heuristic-Search Approaches for the Multi-Objective Shortest-Path Problem: Progress and Research Opportunities  (very good review)
2023 Solving the multi-objective bike routing problem by meta heuristic algorithm (propsed the NGA-MOBRP: new genetic approach for Multi-Objective Bike Routing Problem)
2022 A review and evaluation of multi and many-objective optimization: Methods and algorithms 
2022 Application of state‑of‑the‑art multiobjective metaheuristic algorithms in reliability‑based design optimization: a comparative study (good reveiw but I did not download it because no best algorithm is found)
+ older 2011: Multiobjective evolutionary algorithms: A survey of the state of the art; 2015: Many-objective evolutionary algorithms: A survey  (veyr god to see who cite them ..)
and a very good one is the specific for bike problem: heuristic-enabled Dijkstra algorithm developed by Hrncir et al. (with improved heuristic using ellipse ditance for nodes and epsilon dominance)
and may article on cycle-tourism like "2023 Sustainable cycle-tourism for society: Integrating multi-criteria decision-making and land use approaches for route selection"
or "2022 Multi-objective Route Planning Problem for Cycle-tourists" (good intro for cycling but propose the Augmented P-constraint method (AUGMECON) which looks quite poor)
and many others "2014 A Survey on Algorithmic Approaches for Solving Tourist Trip Design Problem (TTDP)" (highlight the Iterated Local Search (ILS)).
Among the tourist road a recent nice one is "2024 Your trip, your way: An adaptive tourism recommendation system " 
which discuss choice to make etc.. and propose  an  hybrid optimization structure combining Particle Swarm Optimization (PSO) and Differential Evolution Algorithm (DEA).
See alos the "2024 Proposal_of_Hiking_Route_Planning_Optimization_with_Iterated_Local_Search_and_Modified_Tourist_Trip_Design_Problem.pdf" 
which has a goal very close to ours. See also "2012 Route Planning for Bicycles— Exact Constrained Shortest Paths Made Practical Via Contraction Hierarchy (CSP-CH)"


and the pioneerd one "Algorithms for finding paths with multiple constraints"
see also other reviews as "Exact algorithms for multiobjective linear optimization problems with integer variables: A state of the art survey" for integer variables or for Boolean variables: https://doi.org/10.1016/j.cor.2023.106153)
or other models like LLM + "2022: Deep Learning for Trajectory Data Management and Mining: A Survey and Beyond"


Some interesting ideas like in 2021 Most Diverse Near-Shortest Paths: adding adaptative multiplicative  penalties to the edges of already found paths, encouraging the discovery of new, diverse paths 

Anotehr very appealing idea is to perform a huge preprocessing (for instance with an all to all solution or to create the forest hop labeling (FHL) cube) and to use Multi-Objective Dynamic Shortest Path (MODSP) algorothms that 
updating vertices and edges dynamically, querying approximate Pareto fronts, and finding optimal paths based on decision variables and mroe important based on previous results witout aclculatin all (cf review 2022 https://doi.org/10.3390/a16030162)
cf also 10.3233/FAIA240145  or /10.1007/978-3-031-30675-4_15

In python: tons of codes (not laking about the general optmization:  SciPy.optimize  pyOpt, Pyomo): cspy, PyGMO (much better is https://esa.github.io/pygmo2/), pyMCMA, GPOL, pyMultiobjective ( a very good one with all references), paretoset, pathwyse (in C++)... BUt the best one seems to be:
* DEAP (Distributed Evolutionary Algorithms in Python) with Multi-objective optimisation (NSGA-II, NSGA-III, SPEA2, MO-CMA-ES): https://github.com/DEAP/deap or https://pypi.org/project/deap/
* Platypus NSGA-II, NSGA-III, MOEA/D, IBEA, Epsilon-MOEA, SPEA2, GDE3, OMOPSO, SMPSO, and Epsilon-NSGA-II: https://github.com/Project-Platypus/Platypus  https://pypi.org/project/Platypus-Opt/    https://platypus.readthedocs.io/en/latest/
* pymoo: Multi-objective Optimization in Python  : GA, DE, BRKGA, NelderMead, PatternSearch, CMAES, ES, SRES, ISRES, NSGA-II, R-NSGA-II, NSGA-III, U-NSGA-III, R-NSGA-III, MOEAD, AGE-MOEA, C-TAEA, SMS-EMOA, and RVEA.  https://pymoo.org/
+ interesting to test
https://zenodo.org/records/7702018 (Targeted Multiobjective Dijkstra Algorithm + NAMOA_lazy + ..) in C++
https://bitbucket.org/s-ahmadi (for insatnce with the NWMOA* that seems to be the best algorithm for the Exact Multi-objective Path Finding)



IN SUMMARY:  be careful  to choose the best lexicographic order for the cost: this has impact (cf 2023 Heuristic) 
* NGA-MOBRP: (2023 article) is the most suitable to be employed within a real time tool for cyclists: good quality metrics in a reasonable computational time. Slightly faster than Multi-Objective
Simulated Annealing (MOSA) approach
* NWMOA* (2024 Exact..) an exact MOSP algorithm. Way beter than T-MDA: Targeted Multiobejctive Dijkstra algorithm, that was similar to NAMOA∗ dr-lazy to find minimal pareto set
From (mail it seems also faster than ERCA*: (2023 A New Approach for the Resource Constrained Shortest Path Problem): Enhanced Resource Constrained A* (ERCA*): way faster than BiPulse, an existing leading algorithm for RCSPP cf https://github.com/rap-lab-org/public_erca

* WC-EBBA*par (enhanced biased bidirectional A*: paralelism for multi core) (2024 enhanced) is a great choice for applications that need fast solution approaches;
    it is based on RC-BDA* and BiPulse and is bidirectioal,  contrary to the single directional: WC-A*  cf https://bitbucket.org/s-ahmadi/biobj/src/master/

 * Iterated Local Search (ILS) based algotihms such as 2023 Fast approximate bi‑objective Pareto sets .. + Chord algorithm  (to be checked but 10 second for 150 pairs) used in (2024 Proposal of Hiking Route Planning
Optimization with Iterated Local Search and Modified Tourist Trip Design Problem))

* Forest Hop Labeling (FHL) is the only CSP algorithm that can achieve both accurate and efficient results andd all articel by the Xiaofang Zhou's group 
10.1109/ICDE60146.2024.00322 with the exact forest hop labeling (FHL) -cube or better approximate alpha-FHL : 2024 Approximate Skyline Index for Constrained Shortest Pathfinding with Theoretical Guarantee

See also the above one such as multi-objective evolutionary algorithms (MOEAs) and
*  the ε constrained method and Adaptive operator selection (AOS) are used in Multiobjective evolutionary algorithm based on decomposition (MOEA/D) (2014)
indeed: ε-MOEA has been successful in finding well-converged and well-distributed solutions with a much smaller computational effort 
than a number of state-of-the-art MOEAs including NSGA-II, SPEA2, and PESA (as quoted by Kalyanmoy Deb)
see for these approximation an (old review 2021: Approximation Methods for Multiobjective Optimization Problems: A Survey) stressing the difference betwwen 
minimizing and maximizing or between Multiobjective Shortest Path, Spanning Tree (cf  Prim's/Kruskal's Algorithms), Matching, Salesmanor Knapsack Problemas.
It mention a good algorithm by "Approximating Multiobjective Shortest Path in Practice" with teh conclusion than " approximate methods are useful on hard instances with conflicting objectives.
On easier instances, state-of-the-art approximations are not competitive to exact methods"


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


