import heapq
from collections import defaultdict, deque

'''

Exact Multi-objective Path Finding with Negative Weights
Saman Ahmadi1, Nathan R. Sturtevant2, Daniel Harabor3, Mahdi Jalili1

Seems to be the best algorythm for Negative Weight Multi-Objective A* = NWMOA* 



Label Class: Represents a node label with its costs and associated vertex.
NWMOA Class: Implements the NWMOA* algorithm.
__init__ initializes parameters and starts the search.
BFS performs a breadth-first search to find reachable vertices.
dijkstra computes lower bounds using Dijkstra's algorithm.
find_max_delta_f finds the largest delta f-value for the priority queue.
CyclicBucketQueue implements a cyclic bucket queue for the priority queue.
multi_search performs the main multi-objective A* search.
dominates checks if one label dominates another.
is_dominated checks if a label is dominated by any in the expanded list.
remove_dominated removes dominated labels from the expanded list.
'''

class Label:
    def __init__(self, f_values, vertex, incoming_edge=None, path_id=None):
        self.f_values = f_values
        self.vertex = vertex
        self.incoming_edge = incoming_edge
        self.path_id = path_id

    def get_id(self):
        return self.vertex

    def get_f(self):
        return self.f_values

    def __lt__(self, other):
        return self.f_values < other.f_values

class NWMOA:
    def __init__(self, G, G_rev, exp):
        self.num_objs = min(len(G[0]), DIM)
        self.num_vertices = len(G)
        self.h = [[float('inf')] * self.num_objs for _ in range(self.num_vertices)]
        self.Expanded_labels_tr = [[] for _ in range(self.num_vertices)]
        self.Last_label_tr = [None] * self.num_vertices
        self.Sol_set = []
        self.BFS_f = [float('inf')] * self.num_vertices
        self.Paths = defaultdict(list)
        self.label_pool = []
        self.init_search(G, G_rev, exp)

    def init_search(self, G, G_rev, exp):
        start_vertex = exp['start']
        goal_vertex = exp['goal']

        self.BFS(G, start_vertex)
        
        has_neg_cycle = False
        for obj_index in range(self.num_objs):
            has_neg_cycle = self.dijkstra(G_rev, goal_vertex, obj_index)
            if has_neg_cycle:
                print(f"Negative cycle found on dimension {obj_index}")
                return

        if has_neg_cycle:
            return

        max_delta_f = [self.find_max_delta_f(G, obj_index) for obj_index in range(self.num_objs)]
        
        bucket_width = 1
        open_queue = self.CyclicBucketQueue(bucket_width, self.h[start_vertex][0], max_delta_f[0])
        
        total_comp = 0
        self.multi_search(G, open_queue, start_vertex, goal_vertex, total_comp)

    def BFS(self, G, start_vertex):
        queue = deque([start_vertex])
        self.BFS_f[start_vertex] = 0
        while queue:
            current = queue.popleft()
            for neighbor, _ in G[current]:
                if self.BFS_f[neighbor] == float('inf'):
                    self.BFS_f[neighbor] = self.BFS_f[current] + 1
                    queue.append(neighbor)

    def dijkstra(self, G, goal_vertex, obj_index):
        distances = [float('inf')] * self.num_vertices
        distances[goal_vertex] = 0
        pq = [(0, goal_vertex)]
        while pq:
            current_distance, current_vertex = heapq.heappop(pq)
            if current_distance > distances[current_vertex]:
                continue
            for neighbor, weights in G[current_vertex]:
                distance = current_distance + weights[obj_index]
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))
        self.h = distances
        return False  # No negative cycle detection in this simple dijkstra

    def find_max_delta_f(self, G, obj_index):
        max_delta_f = 0
        for u in range(self.num_vertices):
            for v, weights in G[u]:
                max_delta_f = max(max_delta_f, weights[obj_index])
        return max_delta_f

    class CyclicBucketQueue:
        def __init__(self, bucket_width, min_f_value, max_delta_f):
            self.bucket_width = bucket_width
            self.min_f_value = min_f_value
            self.max_delta_f = max_delta_f
            self.buckets = defaultdict(list)
            self.current_min_bucket = 0

        def push(self, label):
            f1_value = label.get_f()[0]
            bucket_index = (f1_value - self.min_f_value) // self.bucket_width
            self.buckets[bucket_index].append(label)

        def pop(self):
            while not self.buckets[self.current_min_bucket]:
                self.current_min_bucket += 1
            return self.buckets[self.current_min_bucket].pop()

        def size(self):
            return sum(len(bucket) for bucket in self.buckets.values())

    def multi_search(self, G, open_queue, start_vertex, goal_vertex, total_comp):
        initial_label = Label(self.h[start_vertex], start_vertex)
        open_queue.push(initial_label)

        while open_queue.size():
            current_label = open_queue.pop()
            current_vertex = current_label.get_id()
            current_label_f = current_label.get_f()
            current_label_tr = tuple(current_label_f[1:])

            if (self.Last_label_tr[current_vertex] and 
                self.dominates(self.Last_label_tr[current_vertex], current_label_tr)):
                continue

            if (self.Last_label_tr[goal_vertex] and 
                self.dominates(self.Last_label_tr[goal_vertex], current_label_tr)):
                continue

            if self.is_dominated(current_label_tr, self.Expanded_labels_tr[current_vertex], total_comp):
                continue

            self.remove_dominated(current_label_tr, self.Expanded_labels_tr[current_vertex])
            self.Expanded_labels_tr[current_vertex].append(current_label_tr)
            self.Last_label_tr[current_vertex] = current_label_tr

            if current_vertex == goal_vertex:
                self.Sol_set.append(current_label)
                continue

            current_label_g = [current_label_f[i] - self.h[current_vertex][i] for i in range(self.num_objs)]

            for edge_id, (tail, edge_weights) in enumerate(G[current_vertex]):
                if self.h[tail][0] == float('inf'):
                    continue
                costs_tail = [current_label_g[i] + edge_weights[i] + self.h[tail][i] for i in range(self.num_objs)]
                tail_label_tr = tuple(costs_tail[1:])
                if (self.Last_label_tr[tail] and 
                    self.dominates(self.Last_label_tr[tail], tail_label_tr)):
                    continue

                new_label = Label(costs_tail, tail, edge_id, current_label.path_id)
                open_queue.push(new_label)

    def dominates(self, label1, label2):
        return all(x <= y for x, y in zip(label1, label2))

    def is_dominated(self, new_label_tr, expanded_labels_tr, comp):
        for label in expanded_labels_tr:
            if self.dominates(label, new_label_tr):
                return True
        return False

    def remove_dominated(self, new_label_tr, expanded_labels_tr):
        expanded_labels_tr[:] = [label for label in expanded_labels_tr if not self.dominates(new_label_tr, label)]

# Example usage
if __name__ == "__main__":
    # Example graph as adjacency list. Each edge has a list of weights for different objectives.
    G = [
        [(1, [1, 2]), (2, [2, 3])],
        [(2, [1, 1]), (3, [4, 5])],
        [(3, [1, 1])],
        []
    ]

    G_rev = [
        [],
        [(0, [1, 2])],
        [(0, [2, 3]), (1, [1, 1])],
        [(1, [4, 5]), (2, [1, 1])]
    ]

    experiment = {'start': 0, 'goal': 3}
    DIM = 2  # number of objectives

    nwmoa = NWMOA(G, G_rev, experiment)
    for solution in nwmoa.Sol_set:
        print("Solution:", solution.get_f())
