#include <queue>
//////////////////////////////////////////////////////
// Breadth First Search
void BFS(graph *G, sn_id_t *h, sn_id_t source)
{
    vertex_deg_t *Out_deg = G->Out_deg;
    edge_data_t Edge_data = G->Edge_data;

    std::queue<sn_id_t> Queue;
    h[source] = 0;
    Queue.push(source);
    
    while (!Queue.empty())
    {
        sn_id_t current_vertex = Queue.front();
        Queue.pop();
                
        for (vertex_deg_t edge_id = 0; edge_id < Out_deg[current_vertex]; edge_id++)
        {
            edge_t edge_data = Edge_data[current_vertex][edge_id];
            sn_id_t tail = edge_data.tail;

            if (h[tail] == SN_ID_MAX)
            {
                h[tail] = h[current_vertex] + 1;
                Queue.push(tail);
            }
        
        }
    }    
}
//////////////////////////////////////////////////////
// Dijkstra's ALgorithm with re-expansions allowed
// The function prunes vertices not reachable from the opposite direction through the "depth_op" array
// it also calculates the non-primary costs of paths through the "ub" array
bool 
Dijkstra(graph *G, pqueue &Queue
    , cost_t **h
    , cost_t **ub
    , sn_id_t *depth_op
    , sn_id_t *parent, sn_id_t source
    , dim_t cost_indx)
{
    vertex_deg_t *Out_deg = G->Out_deg;
    edge_data_t Edge_data = G->Edge_data;

    sn_id_t max_depth = G->Num_vertices;

    sn_id_t *depth = new sn_id_t[G->Num_vertices]();
    
    // Reset + Setup Priority Queue
    Queue.clear();
    Queue.set_cost_index(cost_indx);
    Queue.set_cost_array(h);
    h[source][cost_indx] = 0;
    depth[source] = 0;
    if(ub)
        for(dim_t cost_indx2 = 1; cost_indx2 < DIM; cost_indx2++)
            ub[source][cost_indx2] = 0;
    if(parent) parent[source] = SN_ID_MAX;
    Queue.push(source);
    
    while (Queue.size() > 0)
    {
        sn_id_t current_vertex = Queue.pop();
        
        if (depth[current_vertex] + depth_op[current_vertex] >=  max_depth) return true;
        
        for (vertex_deg_t edge_id = 0; edge_id < Out_deg[current_vertex]; edge_id++)
        {
            edge_t edge_data = Edge_data[current_vertex][edge_id];
            sn_id_t tail = edge_data.tail;

            if (depth_op[tail] == SN_ID_MAX) continue;

            cost_t g1_tail = h[current_vertex][cost_indx] + edge_data.costs[cost_indx];

            if (g1_tail < h[tail][cost_indx])
            {
                h[tail][cost_indx] = g1_tail;
                depth[tail] = depth[current_vertex] + 1;
                if(parent) parent[tail] = current_vertex;
                if (ub)
                    for(dim_t cost_indx2 = 1; cost_indx2 < DIM; cost_indx2++)
                        ub[tail][cost_indx2] = ub[current_vertex][cost_indx2] + edge_data.costs[cost_indx2];

                if (Queue.contains(tail))
                    Queue.decrease_key(tail);
                else
                    Queue.push(tail);
            }
        }
    }

    delete [] depth;
    return false;
    
}
//////////////////////////////////////////////////////
// Bellman-Ford Algorithm
bool 
Bellman_Ford(graph *G, cost_t **h, sn_id_t *depth_op, sn_id_t *parent, sn_id_t source, dim_t cost_indx)
{
    vertex_deg_t *Out_deg = G->Out_deg;
    edge_data_t Edge_data = G->Edge_data;
    
    h[source][cost_indx] = 0;
    if(parent) parent[source] = SN_ID_MAX;
    sn_id_t iteration = 0;
    bool change_observed = false;
    while (iteration <= G->Num_vertices)
    {
        iteration++;
        change_observed = false;
        for (sn_id_t current_vertex = 0; current_vertex < G->Num_vertices; current_vertex++)
        {
            for (vertex_deg_t edge_id = 0; edge_id < Out_deg[current_vertex]; edge_id++)
            {
                edge_t edge_data = Edge_data[current_vertex][edge_id];
                sn_id_t tail = edge_data.tail;
                
                if (depth_op[tail] == SN_ID_MAX || h[current_vertex][cost_indx] == COST_MAX) continue;
                
                if (h[current_vertex][cost_indx] + edge_data.costs[cost_indx] < h[tail][cost_indx])
                {
                    h[tail][cost_indx] = h[current_vertex][cost_indx] + edge_data.costs[cost_indx];
                    if(parent) parent[tail] = current_vertex;
                    change_observed = true;
                }
            }
        }
        if (!change_observed) break;
    }
    if (change_observed) return true;
    return false;
    
}
//////////////////////////////////////////////////////
// Bellman-Ford algorithm with Moore's improvement
// If no source vertex is provided, we treat it as a function to compute potentials for Jhonson's algorithm
bool 
Bellman_Ford_Moore(graph *G, cost_t **h, sn_id_t *depth_op, sn_id_t *parent, sn_id_t source, dim_t cost_indx)
{
    vertex_deg_t *Out_deg = G->Out_deg;
    edge_data_t Edge_data = G->Edge_data;

    sn_id_t vertices_current_iter = 0;
    sn_id_t vertices_next_iter = 0;
    sn_id_t num_iter = 0;

    bool *InQueue = new bool[G->Num_vertices]();
    std::queue<sn_id_t> Queue;
    
    if (source == SN_ID_MAX) // This means we intend to find potentials for Johnson's algorithm
    {
        for (sn_id_t vertex = 0; vertex < G->Num_vertices; vertex++)
        {
            h[vertex][cost_indx] = 0;
            Queue.push(vertex);
            InQueue[vertex] = true;
            vertices_current_iter += 1;
        }
    }
    else
    {
        h[source][cost_indx] = 0;
        if(parent) parent[source] = SN_ID_MAX;
        Queue.push(source);
        InQueue[source] = true;
        vertices_current_iter = 1;
    }
    
    while(!Queue.empty())
    {
        if(vertices_current_iter == 0)
        {
            vertices_current_iter = vertices_next_iter;
            vertices_next_iter = 0;
            num_iter++;
        }

        sn_id_t current_vertex = Queue.front();
        Queue.pop();
        InQueue[current_vertex] = false;
        
        if (num_iter >=  G->Num_vertices) return true;
        
        for (vertex_deg_t edge_id = 0; edge_id < Out_deg[current_vertex]; edge_id++)
        {
            edge_t edge_data = Edge_data[current_vertex][edge_id];
            sn_id_t tail = edge_data.tail;

            if (depth_op[tail] == SN_ID_MAX) continue;

            cost_t g_tail = h[current_vertex][cost_indx] + edge_data.costs[cost_indx];

            if (g_tail < h[tail][cost_indx])
            {
                h[tail][cost_indx] = g_tail;
                if(parent) parent[tail] = current_vertex;
                
                if (!InQueue[tail])
                {
                    Queue.push(tail);
                    vertices_next_iter++;
                    InQueue[tail] = true;
                }
            }
        }
        vertices_current_iter--;
    }

    delete [] InQueue;
    return false;
    
}
//////////////////////////////////////////////////////
// Graph reformulation of Johnson's algorithm 
void 
Reweight_Graph(graph *G, graph *G_rev, cost_t **h, dim_t num_objs)
{
    vertex_deg_t *Out_deg = G->Out_deg;
    edge_data_t Edge_data = G->Edge_data;
    edge_data_t Edge_data_rev = G_rev->Edge_data;

    for (sn_id_t current_vertex = 0; current_vertex < G->Num_vertices; current_vertex++)
    {
        for (vertex_deg_t edge_id = 0; edge_id < Out_deg[current_vertex]; edge_id++)
        {
            edge_t edge_data = Edge_data[current_vertex][edge_id];
            sn_id_t tail = edge_data.tail;
            for (dim_t obj_index = 0; obj_index <num_objs; obj_index++)
            {
                assert(Edge_data[current_vertex][edge_id].costs[obj_index] + h[current_vertex][obj_index] - h[tail][obj_index] >= 0);

                Edge_data[current_vertex][edge_id].costs[obj_index] += h[current_vertex][obj_index] - h[tail][obj_index];
                Edge_data_rev[tail][edge_data.tail_incoming].costs[obj_index] += h[current_vertex][obj_index] - h[tail][obj_index];
                
            }
            
        }
    }
}
//////////////////////////////////////////////////////
// This function finds the maximum Delta f-value in any iteration of A*'s search based on the hueristic function h
// ofr the given index "cost_indx"
void 
Find_max_delta_f(graph *G, cost_t **h, std::array<cost_t, DIM> &max_delta_f, dim_t cost_indx)
{
    vertex_deg_t *Out_deg = G->Out_deg;
    edge_data_t Edge_data = G->Edge_data;
    
    // reset max delta f-value
    max_delta_f[cost_indx] = 0;
    
    for (sn_id_t current_vertex = 0; current_vertex < G->Num_vertices; current_vertex++)
    {
        for (vertex_deg_t edge_id = 0; edge_id < Out_deg[current_vertex]; edge_id++)
        {
            edge_t edge_data = Edge_data[current_vertex][edge_id];
            sn_id_t tail = edge_data.tail;

            if (h[tail][cost_indx] == COST_MAX) continue;

            cost_t delta_f = h[tail][cost_indx] - h[current_vertex][cost_indx] + edge_data.costs[cost_indx];

            if (max_delta_f[cost_indx] < delta_f)
            max_delta_f[cost_indx] = delta_f;
        }
    }
    return;
}
//////////////////////////////////////////////////////




