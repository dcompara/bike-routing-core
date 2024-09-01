#include <iomanip>
#include <algorithm>
#include <iostream>
#include <fstream>

#define PATH // If you want to have all the detail for the paths

///////////////////////////////////
template<class LABEL>
class results_multiobj
{
public:
    results_multiobj() {};
    double_t time_elapsed_init_sec;
    double_t time_elapsed_search_sec;
    std::array<cost_t, DIM> upper_bounds;
    size_t num_sols_b = 0;
    size_t num_sols_f = 0;
    size_t memory_KB;
    std::vector<std::array<cost_t, DIM>> solutions_costs;
    std::vector<std::vector<sn_id_t>> solutions_verticess;
    bool is_rcsp = false;

    void print_stats(experiment exp)
    {
        // priniting stats
        std::cout << std::fixed;
        std::cout.precision(0);
        if (!exp.suppress_header)
        {
            if (is_rcsp)
                std::cerr << "alg\t  queue  map  start_id    goal_id   tightness   opt_cost   #sol   init_time(s)  search_time(s)   search_malloc(KB)\n";
            else
                std::cerr << "alg\t  queue  map  start_id    goal_id   #sol   init_time(s)  runtime(s)   search_malloc(KB)\n";
        }
        std::cout << exp.alg_name << " " << exp.queue_type << "  " << exp.map_name << " "
                  << std::setw(8) << exp.start<< "  "
                  << std::setw(8) << exp.goal<< "  ";

        if (is_rcsp)
        {
            std::cout << exp.constraint << "  " << upper_bounds[0] << "  ";
            // for (uint8_t obj_index = 0; obj_index < 1; obj_index++)
            //   std::cout << std::setw(9) << upper_bounds[obj_index] << "  ";

        }

        std::cout
        // << std::setw(9) << num_sols_f << "  " << std::setw(9) << num_sols_b << "  "
                << std::setw(9) << num_sols_f + num_sols_b << "  ";


        std::cout.precision(6);
        std::cout
                << std::setw(9) << time_elapsed_init_sec << "  "
                << std::setw(9) << time_elapsed_search_sec << "  ";
        std::cout.precision(0);
        std::cout
                << std::setw(9) << memory_KB
                << std::endl
                ;
    }
    ////////////////////////////////////////
    virtual void
    print_paths()
    {
        if(is_rcsp)
        {
            std::cout << "Resource budgets: (" << upper_bounds[1];
            for (uint8_t obj_index = 2; obj_index < DIM; obj_index++)
            {
                std::cout << ", " <<upper_bounds[obj_index] ;
            }
            std::cout << ")" << std::endl;
        }



        // Determine whether to output to a file or to the console
        bool output_to_file = true; // Set this to true or false as needed
        std::ostream* out;
        std::ofstream outFile;

        if (output_to_file)
        {
            outFile.open("output.txt");  // Open the file if needed
            out = &outFile;              // Point to the file stream
        }
        else
        {
            out = &std::cout;            // Point to the console output stream
        }



        // Now print the path details
        for (size_t index = 0; index < solutions_costs.size(); index++)
        {
            *out << std::setfill('-') << std::setw(80) << "-" << std::endl;
            *out << "Path #" << index + 1;
            *out << " costs: (" << (solutions_costs.at(index))[0];
            for (uint8_t obj_index = 1; obj_index < DIM; obj_index++)
            {
                *out << ", " << (solutions_costs.at(index))[obj_index] ;
            }
            *out << ")" << std::endl;

#ifdef PATH
            std::vector<sn_id_t> path_vertices = solutions_verticess.at(index);
            *out << "Full path with " << path_vertices.size() << " vertices -> ";
            *out << "[" << path_vertices[0];
            for (uint i = 1; i < path_vertices.size(); i++)
            {
                *out << "," << path_vertices[i];
            }
            *out << "]" << std::endl;
#endif
        }
#ifndef PATH
        {
            std::cerr<<"Compile with 'make path' for path details. Or simply add #define PATH in the results.h.\n";
        }
#endif

 // Close the file if outputting to a file
        if (output_to_file)
        {
        outFile.close();
    }

    }
    ////////////////////////////////////////
    void store_paths(graph *G_rev, linkedlist<LABEL*> Solutions, Parent_list *Paths, std::string dir)
    {
        edge_data_t Edge_data = G_rev->Edge_data;

        LABEL *solution = Solutions.front();
        while (solution)
        {
            // first recover the first segement via the solution label
            std::vector<sn_id_t> path_vertices;

            sn_id_t current_vertex = solution->get_id();
            path_arr_size path_id = solution->get_path_id();
            vertex_deg_t incoming_link = solution->get_incoming_edge();

            path_vertices.push_back(current_vertex);

            // just follow the path ids stored in the arrays
            while (incoming_link != DEG_MAX)
            {

                edge_t edge_data = Edge_data[current_vertex][incoming_link];

                // Reading from the node of opposite side
                current_vertex = edge_data.tail;
                incoming_link = Paths[current_vertex][path_id].first;
                path_id = Paths[current_vertex][path_id].second;

                path_vertices.push_back(current_vertex);
            }

            // reverse vertex orders for the backward direction
            if (dir == "forward")
            {
                std::reverse(path_vertices.begin(), path_vertices.end());
            }

            // extracting the f-values from the label
            std::array<cost_t, DIM> solution_cost = solution->get_f();

            solutions_costs.push_back(solution_cost);
            solutions_verticess.push_back(path_vertices);

            solution = solution->get_next();
        }

        // reorder solutions
        // if (dir == "forward")
        // {
        //     std::reverse(solutions_verticess.begin(), solutions_verticess.end());
        //     std::reverse(solutions_costs.begin(), solutions_costs.end());
        // }
    }
};
////////////////////////////////////////////////////////////
