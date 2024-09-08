#ifndef DCMOA_H
#define DCMOA_H
// doubly constrained multi-objective shortest path problem with negative weights
// based on DCMOA
// @author: Daniel COMPARAT
// created: 02/09/2024
// updated: 02/09/2024

/******************************************************

Sketch of DCMOA Algorithm

1.	Initialization:
o	Input: Graph G, reversed graph G_rev, start vertex, goal vertex, resource constraints (budgets), number of objectives.
o	Initialize:
    	Cost vectors (heuristic h and upper bounds ub).
    	Resource budgets based on the input constraints.
    	Priority queue Open.
    	Data structures for expanded labels and solution sets.

2.	Preprocessing:
o	Perform a Breadth-First Search (BFS) to find reachable vertices from the start.
o	Calculate lower bounds for each objective using Dijkstra's algorithm (and upper bounds for non-primary objectives).
o	If a negative cycle is detected, terminate the algorithm as the problem is unsolvable.
o	Adjust resource budgets based on the calculated upper bounds and the given tightness constraints.

3.	Main Search Loop:
o	While Open is not empty:
1.	Extract the node with the smallest f1 value from Open.
2.	Budget Check: If the primary cost exceeds the budget, terminate the search early.
3.	Quick Dominance Check: Check if the node is dominated by the most recently expanded node or if it violates resource budgets. If dominated or violating, discard it.
4.	Full Dominance Check: Compare the current node against all previously expanded nodes to ensure it's not dominated.
5.	If the node is non-dominated, expand it:
    	   Generate new nodes (successors).
    	Calculate their cost vectors.
    	Perform dominance and budget checks before adding them to the queue.
6.	If the node corresponds to the goal:
    	Update the budget with the cost of the solution.
    	Add it to the solution set and check for any dominated solutions in the set.

4.	Termination:
o	The algorithm terminates when the goal is reached and a valid (within budget) path is found.
o	Output: The shortest path that satisfies all resource constraints.



******************************************************/

template <class LABEL, class Q, class H>
class DCMOA
{

public:
    DCMOA(graph *G, graph *G_rev, experiment exp) : LABEL_Pool_(1024)
    {
        // retrieve goal and start vertex
        sn_id_t start_vertex = exp.start - G->Vertex_offset;
        sn_id_t goal_vertex = exp.goal - G->Vertex_offset;

        // Initialise necessary parameters and data structures
        initialise_parameters(G, exp);


        // Negative Cycle flag
        bool has_neg_cycle = false;

        ////////////////////////////////////////////////////
        // Here we do some quick calculations to come up with resource budgets, subject to not having negative cycles
        // Note: This section can be skipped if the resource budgets are already known

        // First, find reachable vertices from start, and their distances
        BFS(G, BFS_f, start_vertex); // Forward Breadth First Search

        // Calculate lower bounds and check negative cycles
        for (dim_t obj_index = 0; obj_index <num_objs; obj_index++)
        {
            // Run backward Dijkstra's algorithm with re-expansions allowed to calculate lower bounds and detect negative cycles
            if(obj_index != 0) // we find some upper bound on non-primary costs (resources) through 'ub'
                // DEBUG It find the lowest. So is it really upper bound??
                has_neg_cycle =  Dijkstra(G_rev, *Open_1, h_, ub_, BFS_f, NULL, goal_vertex, obj_index);
            else
                has_neg_cycle = Dijkstra(G_rev, *Open_1, h_, NULL, BFS_f, NULL, goal_vertex, obj_index);

            if (has_neg_cycle)
            {
                std::cerr << "Negative cycle found on dimension " << int(obj_index) << std::endl;
                break;
            }
        }

        // The instance cannot be solved by DCMOA*
        if (has_neg_cycle) return;

        // let's setup budget resources based on the tightness given by exp.constraint;
        for (dim_t obj_index = 1; obj_index <num_objs; obj_index++)
        {
            budgets_[obj_index] = std::floor(exp.constraint*(ub_[start_vertex][obj_index] - h_[start_vertex][obj_index])/100) + h_[start_vertex][obj_index];
            // budgets_[obj_index] = 11000;
            std::cout << "DEBUG to be modified " << std::endl;
            std::cout << "upper bound [" << int(obj_index) << "] = " << ub_[start_vertex][obj_index]  << std::endl;
            std::cout << "h [" << int(obj_index) << "] = " << h_[start_vertex][obj_index]  << std::endl;
            std::cout << "constraint = " << exp.constraint  << std::endl;
            std::cout << "budgets_[" << int(obj_index) << "] = " << budgets_[obj_index]  << std::endl;
        }

        // Reset lower bounds and BFS array for the upcoming searches
        // DEBUG: is it needed ???
        for (sn_id_t id = 0; id < G->Num_vertices; id++)
        {
            {
                for (dim_t obj_index = 0; obj_index <num_objs; obj_index++) h_[id][obj_index] = COST_MAX;
            }
            BFS_f[id] = SN_ID_MAX;
        }

        ////////////////////////////////////////////////////
        // start from scratch the preliminary searches needed to calculate lower bounds
        // Turn on the timer
        timer mytimer;
        mytimer.start();

        // Again, find reachable vertices from start, and their distances
        BFS(G, BFS_f, start_vertex); // Forward Breadth First Search

        // Calculate lower bounds and check negative cycles
        for (int obj_index = num_objs - 1; obj_index >= 0; obj_index--)
        {
            has_neg_cycle = Dijkstra(G_rev, *Open_1, h_, NULL, BFS_f, NULL, goal_vertex, obj_index);
            // ALTERNATIVE: Run Bellman-Ford to compute lower bounds and detect negative cycles
            // has_neg_cycle = Bellman_Ford_Moore(G_rev, h_, BFS_f, NULL, goal_vertex, obj_index);

            if (has_neg_cycle)
            {
                std::cerr << "Negative cycle found on dimension " << int(obj_index) << std::endl;
                break;
            }
        }


        // The instance cannot be solved by DCMOA*
        if (has_neg_cycle) return;

        // Now find the largest Delta f-value for the nodes in the priority queue in any iteration of A*
        // We find it for the primary cost (index 0) only
        std::array<cost_t, DIM> max_delta_f;
        Find_max_delta_f(G, h_, max_delta_f, 0);

        // Stop the timer of the initilisation phase
        mytimer.stop();
        double time_init = mytimer.elapsed_time_second();
        /////////////////////////////////////////////////////////////////////////////
        // The main search is done here
        // first reset the timer
        mytimer.reset();
        mytimer.start();

        // Initialise the A*'s priority queue based on the provided Delta f-value
        // Priority queue can be a fixed-size cyclic bucket queue where the size of the buckets is set by max_delta_f
        cost_t bucket_width = 1;
        Q Open_2(bucket_width, h_[start_vertex][0], max_delta_f[0]); // it takes (bucket_width, min f-value, max delta f-value)

        // A counter for the number of dominace checks performed
        size_t total_comp = 0;

        // Execute the main A* search
        Multi_search(G, Open_2, LABEL_Pool_, Sol_set_, h_, budgets_, Expanded_labels_tr_, Last_label_tr_, start_vertex, goal_vertex, Paths_, total_comp);




        // Stop timer
        mytimer.stop();
        double time_search = mytimer.elapsed_time_second();

        // Calculate memory used for all labels + priority queue + all stored paths (expanded + backtracking)
        size_t search_mem = LABEL_Pool_.mem() + Open_2.mem() + this->mem();

        // Store search results
        results_multiobj<LABEL> res;
        res.time_elapsed_init_sec = time_init;
        res.time_elapsed_search_sec = time_search;
        res.memory_KB = search_mem/1024;
        res.num_sols_f = Sol_set_.size();
        res.is_rcsp = true;
        for (dim_t obj_index = 0; obj_index < num_objs; obj_index++)
        {
            res.upper_bounds[obj_index] = budgets_[obj_index];
        }

        // Print results
        res.print_stats(exp);

        if(exp.path) // Capture and then print path details if requested
        {
            // recover paths
            res.store_paths(G_rev, Sol_set_, Paths_, "forward");
            res.print_paths();
        }
    }



    ~DCMOA()
    {
        for (sn_id_t id = 0; id < num_vertices; id++)
        {
            delete [] h_[id];
            delete [] ub_[id];
        }
        delete [] BFS_f;
        delete [] Last_label_tr_;
        delete [] Expanded_labels_tr_;
        delete [] h_;
        delete [] ub_;
        delete [] budgets_;
        delete Open_1;

#ifdef PATH
        delete [] Paths_;
#endif
    }

    size_t
    mem()
    {
        size_t bytes = 0;
        for (sn_id_t id = 0; id < num_vertices; id++)
        {
            // memory of backtracking
            if (Paths_)
            {
                bytes += Paths_[id].capacity()*sizeof(std::pair<vertex_deg_t, path_arr_size>);
            }

            // memory of expanded paths
            bytes += Expanded_labels_tr_[id].capacity()*sizeof(LABEL_TR);

        }
        return bytes;
    }



//////////////////////////////////////////////////////
private:
    dim_t num_objs;
    sn_id_t num_vertices;
    cost_t **h_, **ub_;
    cost_t *budgets_;
    pqueue *Open_1;
    label_pool<LABEL> LABEL_Pool_; // initialising the label pool with initial size of 1024
    Parent_list *Paths_;
    LABEL_TR_list *Expanded_labels_tr_;
    LABEL_TR *Last_label_tr_;
    linkedlist<LABEL*> Sol_set_; // Initialising solution sets
    sn_id_t *BFS_f;
//////////////////////////////////////////////////////
    void
    initialise_parameters(graph *G, experiment exp)
    {
        // Get the number of vertices
        num_vertices = G->Num_vertices;
        // Assert the number of objectives
        num_objs = std::min(G->Num_objectives, DIM);
        // initialise lower and upperbound arrays
        h_ = new cost_t*[num_vertices]();
        ub_ = new cost_t*[num_vertices]();
        // Initialise Expanded_labels for truncated cost_vectors of expnaded paths
        Expanded_labels_tr_ = new LABEL_TR_list[num_vertices]();
        // Initialise Last_label for truncated cost_vectors of the last expansion
        Last_label_tr_ = new LABEL_TR[num_vertices]();
        // Intilialise an array to store the results of the forward BFS
        BFS_f = new sn_id_t[num_vertices];
        // An array to store resource budgets
        budgets_ = new cost_t[num_objs]();
        budgets_[0] = COST_MAX;

        // Initialise arrays per vertex
        for (sn_id_t id = 0; id < num_vertices; id++)
        {
            h_[id] = new cost_t[num_objs]();
            ub_[id] = new cost_t[num_objs]();
            for (dim_t obj_index = 0; obj_index <num_objs; obj_index++)
            {
                h_[id][obj_index] = COST_MAX;
            }
            // Initialise min_distances for BFS searches
            BFS_f[id] = SN_ID_MAX;
        }

        // Intialise priority queue of preliminary Dijkstra's search
        Open_1 = new pqueue(1024, h_, num_vertices);

#ifdef PATH
        // Intialise array of expanded paths
        Paths_ = new Parent_list[num_vertices]();
#else
        Paths_ = 0;
#endif

    }


/******************************************

	Main Search Loop:

o	While Open is not empty:
    1.	Extract the node with the smallest f1 value from Open.
    2.	Budget Check: If the primary cost exceeds the budget, terminate the search early.
    3.	Quick Dominance Check: Check if the node is dominated by the most recently expanded node (or if it violates resource budgets) discard it.
    4.	Full Dominance Check: Compare the current node against all previously expanded nodes to ensure it's not dominated.
    5.	If the node is non-dominated, expand it:
        	   Generate new nodes (successors).
        	Calculate their cost vectors.
        	Perform dominance and budget checks before adding them to the queue.
    6.	If the node corresponds to the goal:
        	Update the budget with the cost of the solution.
        	Add it to the solution set and check for any dominated solutions in the set.

*********************************************/


    void *Multi_search(
        graph *G
        , Q &Open
        , label_pool<LABEL> &label_pool
        , linkedlist<LABEL*> &Sol_set
        , cost_t **h
        , cost_t *&budgets
        , LABEL_TR_list *Expanded_labels_tr
        , LABEL_TR* Last_label_tr
        , sn_id_t initial, sn_id_t target
        , Parent_list *&Paths
        // , size_t &generated, size_t &expansions, size_t &pruned, size_t &pruned_last
        , size_t &comp_
    )
    {
        // Load graph data
        vertex_deg_t *Out_deg = G->Out_deg;
        edge_data_t Edge_data = G->Edge_data;

        // generating the first label for the initial node
        LABEL *current_label = label_pool.get_label();
        *current_label = LABEL(h[initial], initial, DEG_MAX, PATH_ARR_SIZE_MAX);

        // Insert the initial lable into the queue
        Open.push(current_label);

        // Initialise a cost array to be used within the search
        std::array<cost_t, DIM> current_label_g;

        // create a truncated label for budgets
        // all valid labels must weakly dominate the budget vector
        LABEL_TR budget_label_tr(budgets);

        // Search while the queue is not empty
        while (Open.size())
        {
            // Extract (pop) the least-cost label (can be non-lexicographical)
            LABEL *current_label = Open.pop();

            // recover vertex_id from the label
            sn_id_t current_vertex = current_label->get_id();

            // extracting the f-values from the label
            std::array<cost_t, DIM> current_label_f = current_label->get_f();

// TODO (Daniel#1#): Not in NWMOA. SO remove for DEBUG ...
            // Termination criterion, stop the search if the current solution is proved to be optimal
//            if (current_label->get_f_pri() > budgets[0])
//            {
//                label_pool.save_label(current_label);
//                // DEBUG
//                std::cout << "  Termination criterion for budget [0] = " << budgets[0] << " current_label->get_f_pri() = " << current_label->get_f_pri() << std::endl;
//                break;
//            }

            if (current_label->get_f_pri() > budgets[0])
            {
                label_pool.save_label(current_label);
                // DEBUG
                std::cout << "  Termination criterion for budget [0] = " << budgets[0] << " current_label->get_f_pri() = " << current_label->get_f_pri() << std::endl;
                continue;
            }


            // Create a truncated label by removing the first element
            LABEL_TR current_label_tr(current_label_f);

// TODO (Daniel#1#): DEBUG add NWMOA
                // Perform Quick Dominance check and prune if the extracted label is dominated
            // the operation "L<<R" means L dominates R
            if (Last_label_tr[current_vertex] << current_label_tr || Last_label_tr[target] << current_label_tr)
            {
                label_pool.save_label(current_label);    // if dominated, thus skip expansion + recycle the label
                continue;
            }


            // Perform Quick Dominance check and prune if the extracted label is dominated
            // the operation "L<<R" means L dominates R
//            if (Last_label_tr[current_vertex] << current_label_tr)
//            {
//                label_pool.save_label(current_label);    // if dominated, thus skip expansion + recycle the label
//                continue;
//            }

            // Perform a linear Dominance check against previous expansion of the vertex in lexicographical order
            std::pair<bool, LABEL_TR_iter> check_result = dominance_check(current_label_tr, Expanded_labels_tr[current_vertex], comp_);
            bool is_dominated = check_result.first;
            if (is_dominated)
            {
                label_pool.save_label(current_label);    // if dominated, thus skip expansion + recycle the label
                continue;
            }

// TODO (Daniel#1#): DEBUG add the second dominance tets as in MWMOA to test if it work the same way
       // Perform a linear Dominance check, this time against the target verex
            std::pair<bool, LABEL_TR_iter> check2_result = dominance_check(current_label_tr, Expanded_labels_tr[target], comp_);
            is_dominated = check2_result.first;
            if (is_dominated == true)
            {
                label_pool.save_label(current_label);
                continue;
            }

            // The extracted label is non-dominated, thus store the location of label in the vector to later add to the expanded list
            LABEL_TR_iter place_to_add = check_result.second;

            // Attempt to remove dominated labels and add the current label to the place already obtained above
            remove_dominated(current_label_tr, Expanded_labels_tr[current_vertex], place_to_add, comp_);
            add_to_expanded(current_label_tr, Expanded_labels_tr[current_vertex], place_to_add);

            // Store the label as the most recent expansion
            Last_label_tr[current_vertex] = current_label_tr;

            // Never expand the target, but capture the solution
            if (current_vertex == target)
            {
                // First, store the cost of the solution
                budgets[0] = current_label->get_f_pri();
                // Since we are not expanding nodes lexicographically, we need to check if
                // the new solution dominates any of the previous solutions
                LABEL* last_sol = 0;
                LABEL* current_sol = Sol_set.front();
                while(current_sol && current_sol->get_f_pri() == current_label->get_f_pri())
                {
                    LABEL* tmp = 0;
                    if (*current_label << (*current_sol)) // the new solution dominates a previous solution
                    {
                        Sol_set.pop_after(last_sol);    // the dominated solution should be removed
                        tmp = current_sol;
                    }
                    else
                        last_sol = current_sol;

                    current_sol = current_sol->get_next(); // try another solution
                    if (tmp) label_pool.save_label(tmp); // recycle the dominated solution label
                }

                Sol_set.push_front(current_label);
                continue; // no need to expand the solution path
            }

            // Get ready for expansion, recover g-values from the label
            for (dim_t i = 0; i < num_objs; ++i)
            {
                current_label_g[i] = current_label_f[i] - h[current_vertex][i];
            }

#ifdef PATH
            // Retrieve next path id
            path_arr_size path_id = Paths[current_vertex].size();
            // Store backtracking information
            // CAUTION:: Max parent_array_id is UINT16_MAX
            Paths[current_vertex].push_back(std::make_pair(current_label->get_incoming_edge(), current_label->get_path_id()));
#endif

            // recycle the label
            label_pool.save_label(current_label);

            // Expand successors
            for (vertex_deg_t edge_id = 0; edge_id < Out_deg[current_vertex]; edge_id++)
            {
                // Retrieve successor vertex and its edge data
                edge_t edge_data = Edge_data[current_vertex][edge_id];
                sn_id_t tail = edge_data.tail;

                // Check if it is possible to get to the target through the successor vertex
                if (h[tail][0] == COST_MAX)
                {
                    continue;
                }

                // Build f-values of the extended path
                std::array<cost_t, DIM> costs_tail;
                for (dim_t i = 0; i < num_objs; ++i)
                {
                    costs_tail[i] = current_label_g[i] +  edge_data.costs[i] + h[tail][i];
                }

                // Create the truncated label of extended path
                LABEL_TR tail_label_tr(costs_tail);

// TODO (Daniel#1#): DEBUG T create articificllai NWMOA
       // Attempt a quick dominance check against two candidates
                if (Last_label_tr[tail] << tail_label_tr || Last_label_tr[target] << tail_label_tr)
                {
                    continue;
                }


//                // Attempt Quick dominance and upperbound checks
//                // every valid label must weakly dominate the budget vector
//                if ( !(tail_label_tr << budget_label_tr) || Last_label_tr[tail] << tail_label_tr)
//                {
//                    continue;
//                }

                // Genereate a new label and put it into the queue
                LABEL *new_label = label_pool.get_label();
#ifdef PATH
                *new_label = LABEL(costs_tail, tail, edge_data.tail_incoming, path_id); // keep backtracking information
#else
                *new_label = LABEL(costs_tail, tail);
#endif

                // Add it to the queue
                Open.push(new_label);
            }

        }

        return NULL;
    }
////////////////////////////////////////////
// This function simply prints the costs stored in the label
    void
    print_label(LABEL *label)
    {
        std::array<cost_t, DIM> label_f = label->get_f();
        for(dim_t i = 0; i < num_objs; i++)
        {
            std::cerr<< label_f[i] << " ";
        }
        std::cerr<<std::endl;
    }
////////////////////////////////////////////
// This function performs a dominance test over a list of (truncated) vectors lexicographically
    std::pair<bool, LABEL_TR_iter>
    dominance_check(LABEL_TR new_label_tr, LABEL_TR_list &Exp_labels_tr, size_t &comp_)
    {
        bool dominated = false;
        LABEL_TR_iter it = Exp_labels_tr.begin();
        while (it < Exp_labels_tr.end())
        {
            // comp_++;
            if (new_label_tr <= (*it))
                break;

            // comp_++;
            if ((*it) << new_label_tr)
            {
                dominated = true;
                break;
            }
            ++it; // Move the iterator to the next element
        }
        return std::make_pair(dominated, it);
    }
////////////////////////////////////////////
// This function iterates backwards through the list and remove (truncated) vectors dominated by "new_label_tr"
    void remove_dominated(const LABEL_TR& new_label_tr, LABEL_TR_list &Exp_labels_tr, LABEL_TR_iter it, size_t &comp_)
    {
        auto reverse_it = Exp_labels_tr.rbegin();

        while (reverse_it != Exp_labels_tr.rend() && reverse_it.base() != it)
        {
            if (new_label_tr << (*reverse_it))
                reverse_it = decltype(reverse_it)(Exp_labels_tr.erase(std::next(reverse_it).base()));
            else
                ++reverse_it; // Move the reverse iterator to the next element
        }
    }
////////////////////////////////////////////
// This function iterates forward through the list and remove (truncated) vectors dominated by "new_label_tr"
    void
    remove_dominated_forward(LABEL_TR new_label_tr, LABEL_TR_list &Exp_labels_tr, LABEL_TR_iter it, size_t &comp_)
    {
        while (it < Exp_labels_tr.end())
        {
            if (new_label_tr << (*it))
                it = Exp_labels_tr.erase(it);
            else
                ++it; // Move the iterator to the next element
        }
    }
////////////////////////////////////////////
// This function just adds the truncated vector "new_label_tr" into the iterator position "it"
    void
    add_to_expanded(LABEL_TR new_label_tr, LABEL_TR_list &Exp_labels_tr, LABEL_TR_iter it)
    {
        Exp_labels_tr.insert(it, new_label_tr);
    }
//////////////////////////////////////////////////////
};
#endif
