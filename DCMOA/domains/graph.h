#ifndef GRAPH_H
#define GRAPH_H
#include "constants.h"
#include <iostream>

class graph
{
public:
    graph():
    Num_vertices(0), Num_edges(0), reverse(0)
    {};

    ~graph()
    {
        delete [] Out_deg;

        if (!reverse)
        {
            delete [] xy_co;
        }
    };

    sn_id_t Num_vertices;
    sn_id_t Num_edges;
    sn_id_t Vertex_offset;
    dim_t Num_objectives;
    bool reverse;

    int16_t *xy_ele;
    coord_pair *xy_co;
    vertex_deg_t *Out_deg;
    edge_data_t Edge_data;
};

void load_graph(graph* G, graph* G_rev, std::string input_file_name);
void write_graph(std::ostream& out, graph *G);
void write_graph_randomized(std::ostream& out, graph *G);
void write_graph_dimacs(std::ostream& out, graph *G, dim_t cost_index);

#endif // GRAPH_H
