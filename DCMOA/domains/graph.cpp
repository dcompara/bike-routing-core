#include <fstream>
#include <sstream>  // Add this line at the top of your file
#include <cassert>
#include "graph.h"
#include "timer.h"


void load_graph(graph *G, graph *G_rev, std::string input_file_name)
{
    sn_id_t num_vertices, num_edges;
    dim_t num_objectives;
    std::ifstream ifs(input_file_name);
    // Read graph info
    while (ifs.good())
    {
        ifs >> std::ws;
        if (ifs.peek() == '#')
        {
            while (ifs.get() != '\n')
                ;
            continue;
        }

        if (ifs.peek() == 'n')
        {
            while (ifs.get() != ' ')
                ;
        } // "nodes" keyword
        ifs >> num_vertices;
        ifs >> std::ws;
        if (ifs.peek() == 'e')
        {
            while (ifs.get() != ' ')
                ;
        } // "edges" keyword
        ifs >> num_edges;
        ifs >> std::ws;
        if (ifs.peek() == 'o')
        {
            while (ifs.get() != ' ')
                ;
        } // "objectives" keyword
        ifs >> num_objectives;
        ifs >> std::ws;
        break;
    }

    int16_t *xy_ele = new int16_t[num_vertices];
    coord_pair *xy_co = new coord_pair[num_vertices];
    vertex_deg_t *Out_deg = new vertex_deg_t[num_vertices](); // the () initialize to 0
    vertex_deg_t *In_deg = new vertex_deg_t[num_vertices](); // the () initialize to 0

    struct edge_full
    {
        sn_id_t head;
        sn_id_t tail;
        vertex_deg_t head_outgoing;
        vertex_deg_t tail_incoming;
        std::vector<edge_cost_t> costs;

    };

    std::vector<edge_full> all_edge_data(num_edges);

    uint32_t v_added = 0, e_added = 0;
    sn_id_t vertex_offset = SN_ID_MAX; // Node identifiers may start from non-zero values
    while (ifs.good())
    {
        // Read vertices
        ifs >> std::ws;
        while (ifs.peek() == 'v')
        {
            sn_id_t id;
            int32_t x, y; // this reads the the first two elements as location (lat,long) of the vertex
            int16_t ele; // this reads the third element as elevation of the vertex
            ifs.get(); // eat the 'v' char
            ifs >> id >> x >> y >> ele;
            if (id < vertex_offset) vertex_offset = id; // assumes the smallest vertex id appears first
            xy_co[id - vertex_offset] = std::make_pair(x,y);
            xy_ele[id - vertex_offset] = ele;
            ifs >> std::ws; // trailing whitespace
            v_added++;
        }

        // if no vertex line found, assume vertices start from 0
        if (vertex_offset == SN_ID_MAX) vertex_offset = 0;

        // Read the edges
        while (ifs.peek() == 'e')
        {
            assert(e_added < num_edges);  // Program will terminate if this condition is false, so if we try to add too much edges

            sn_id_t from_id, to_id;
            edge_cost_t value;
            std::vector<edge_cost_t> costs;

            ifs.get(); // eat the 'e' char
            ifs >> from_id >> to_id;



//            while (ifs.peek() != '\n')
//            {
//                ifs >> value;
//                costs.push_back(value);
//            }


            std::string line;
            std::getline(ifs, line);  // Read the rest of the line
            std::istringstream iss(line);  // Use a stringstream to parse the rest of the line
            while (iss >> value)
            {
                costs.push_back(value);
            }

            // check if the number of objectives is less than what declared
            if (costs.size() < num_objectives) num_objectives = costs.size();

            all_edge_data[e_added]= {from_id - vertex_offset, to_id - vertex_offset, Out_deg[from_id], In_deg[to_id], costs};

            Out_deg[from_id - vertex_offset]++;
            In_deg[to_id - vertex_offset]++;

            e_added++;
            ifs >> std::ws;
        }

        num_objectives = std::min(DIM, num_objectives);

        // Initialize the Succ_edge_data and Pred_edge_data with vectors
        edge_data_t Succ_edge_data(num_vertices);
        edge_data_t Pred_edge_data(num_vertices);

        for (sn_id_t ver = 0; ver < num_vertices; ver++)
        {
            Succ_edge_data[ver].resize(Out_deg[ver]);
            Pred_edge_data[ver].resize(In_deg[ver]);
        }

        for (sn_id_t ed = 0; ed < e_added; ed++)
        {
            edge_full edge_ = all_edge_data[ed];
            std::array<edge_cost_t, DIM> costs_array;
            for (dim_t i = 0; i < num_objectives; ++i)
            {
                costs_array[i] = std::ceil(edge_.costs[i]);
            }

            Succ_edge_data[edge_.head][edge_.head_outgoing] = {edge_.tail, edge_.tail_incoming, costs_array};
            Pred_edge_data[edge_.tail][edge_.tail_incoming] = {edge_.head, edge_.head_outgoing, costs_array};
        }

        // Now build graph G
        G->Num_vertices = num_vertices;
        G->Num_edges = e_added;
        G->Num_objectives = num_objectives;
        G->Vertex_offset = vertex_offset;
        G->reverse = false;
        G->xy_ele = xy_ele;
        G->xy_co = xy_co;
        G->Out_deg = Out_deg;
        G->Edge_data = Succ_edge_data;

        // Then build the reversed graph G_rev
        G_rev->Num_vertices = num_vertices;
        G_rev->Num_edges = e_added;
        G_rev->Num_objectives = num_objectives;
        G_rev->Vertex_offset = vertex_offset;
        G_rev->reverse = true;
        G_rev->xy_ele = xy_ele;
        G_rev->xy_co = xy_co;
        G_rev->Out_deg = In_deg;
        G_rev->Edge_data = Pred_edge_data;

        std::cerr << "Graph loaded with " << v_added << " vertices and " << e_added << " edges and " << num_objectives << " objectives.\n";

        break;
    }
}
//////////////////////////////////////////
/// This function write the graph as an xy graph
void write_graph(std::ostream& out, graph *G)
{
    timer mytimer;
    mytimer.start();
    // comments
    out << "# warthog xy graph\n"
        << "# this file is formatted as follows: [header data] [node data] [edge data]\n"
        << "# header format: nodes [number of nodes] edges [number of edges] objectives [number of objectives]\n"
        << "# node data format: v [id] [x] [y] [elevation]\n"
        << "# edge data format: e [from_node_id] [to_node_id] [cost_1] [cost_2] ... [cost_k]\n"
        << "#\n"
        << "# 32bit integer values are used throughout.\n"
        << "# Identifiers are all zero indexed.\n"
        << "# Node identifiers start from zero.\n"
        << "#\n";

    out<<std::fixed;
    out.precision(0);
    out
            << "nodes " << G->Num_vertices << " edges " << G->Num_edges << " objectives " << G->Num_objectives << std::endl;

    // node data
    for(uint32_t i = 0; i < G->Num_vertices; i++)
    {
        int32_t x = G->xy_co[i].first;
        int32_t y = G->xy_co[i].second;
        int32_t ele = G->xy_ele[i];

        out
                << "v " << i << " "
                << x << " "
                << y << " "
                << ele
                << std::endl;
    }

    out.precision(0);
    for(sn_id_t i = 0; i < G->Num_vertices; i++)
    {
        for(vertex_deg_t edge_idx = 0; edge_idx < G->Out_deg[i]; edge_idx++)
        {
            edge_t edge_data = G->Edge_data[i][edge_idx];
            out << "e " << i << " " << edge_data.tail << " ";
            for (dim_t obj = 0; obj < G->Num_objectives; ++obj)
            {
                out << " " << edge_data.costs[obj];
            }
            out << std::endl;
        }
    }

    mytimer.stop();
    std::cerr
            << "wrote xy_graph; time "
            << ((double)mytimer.elapsed_time_nano() / 1e9)
            << " s" << std::endl;
}


////////////////////////////////////
/// This function turns the input graph into a random graph with negative weights but without negative cycles
void write_graph_randomized(std::ostream& out, graph *G)
{
    timer mytimer;
    mytimer.start();
    // comments
    out << "# warthog xy Randomised graph\n"
        << "# this file is formatted as follows: [header data] [node data] [edge data]\n"
        << "# header format: nodes [number of nodes] edges [number of edges] \n"
        << "# node data format: v [id] [x] [y] [elevation]\n"
        << "# edge data format: e [from_node_id] [to_node_id] [cost_1] [cost_2] ... [cost_k]\n"
        << "#\n"
        << "# 32bit integer values are used throughout.\n"
        << "# Identifiers are all zero indexed.\n"
        << "#\n";

    out<<std::fixed;
    out.precision(0);
    out
            << "nodes " << G->Num_vertices << " edges " << G->Num_edges << " objectives "<<  G->Num_objectives << std::endl;

    // node data
    for(uint32_t i = 0; i < G->Num_vertices; i++)
    {
        int32_t x = G->xy_co[i].first;
        int32_t y = G->xy_co[i].second;
        int32_t ele = G->xy_ele[i];

        out
                << "v " << i << " "
                << x << " "
                << y << " "
                << ele
                << std::endl;
    }

    cost_t** potentials = new cost_t*[G->Num_vertices]();
    for (sn_id_t ver = 0; ver < G->Num_vertices; ver++)
    {
        potentials[ver] = new cost_t[G->Num_objectives]();
        for (dim_t obj = 0; obj < G->Num_objectives; obj++)
            potentials[ver][obj] = (rand() % 100) - 100; // generate random negative potentials
    }

    uint neg_counter = 0;
    cost_t min_cost = 1000;
    cost_t max_cost = -1000;

    cost_t* delta_potential = new cost_t[G->Num_objectives]();
    out.precision(0);
    for(sn_id_t ver = 0; ver < G->Num_vertices; ver++)
    {
        for(vertex_deg_t edge_idx = 0; edge_idx < G->Out_deg[ver]; edge_idx++)
        {
            edge_t edge_data = G->Edge_data[ver][edge_idx];
            for (dim_t obj = 0; obj < G->Num_objectives; obj++)
            {
                delta_potential[obj] = potentials[edge_data.tail][obj] - potentials[ver][obj];
                edge_data.costs[obj] = (rand() % 10) - delta_potential[obj];

            }
            if (edge_data.costs[0] < 0) neg_counter++;
            if (edge_data.costs[0] < min_cost) min_cost = edge_data.costs[0];
            if (edge_data.costs[0] > max_cost) max_cost = edge_data.costs[0];
            out << "e " << ver << " " << edge_data.tail << " ";
            for (dim_t obj = 0; obj < G->Num_objectives; ++obj)
            {
                out << " " << edge_data.costs[obj];
            }
            out << std::endl;
        }
    }

    mytimer.stop();
    std::cerr
            << "wrote xy_graph; time "
            << ((double)mytimer.elapsed_time_nano() / 1e9)
            << " s" << std::endl;
    std::cerr
            << "Percentage of negative weights: "<< 100*neg_counter/G->Num_edges
            << ", cost range: [" << min_cost << ", " <<max_cost << "]\n";

    for (sn_id_t ver = 0; ver < G->Num_vertices; ver++)
    {
        delete [] potentials[ver];
    }
    delete [] potentials;
    delete [] delta_potential;
}
/////////////////////////////////////////
//////////// This function turns the graph into the DIMACS format 1-index for a given index
void write_graph_dimacs(std::ostream& out, graph *G, dim_t cost_index)
{
    if (cost_index > G->Num_objectives) return;
    timer mytimer;
    mytimer.start();

    out<<std::fixed;
    out.precision(0);
    out << "p sp " << G->Num_vertices << " " << G->Num_edges << std::endl;

    out.precision(0);
    for(sn_id_t ver = 0; ver < G->Num_vertices; ver++)
    {
        for(vertex_deg_t edge_idx = 0; edge_idx < G->Out_deg[ver]; edge_idx++)
        {
            edge_t edge_data = G->Edge_data[ver][edge_idx];
            out << "a " << ver + 1 << " " << edge_data.tail + 1 << " " << edge_data.costs[cost_index] << std::endl;
        }
    }

    mytimer.stop();
    std::cerr
            << "wrote DIMACS_graph; time "
            << ((double)mytimer.elapsed_time_nano() / 1e9)
            << " s" << std::endl;
}
