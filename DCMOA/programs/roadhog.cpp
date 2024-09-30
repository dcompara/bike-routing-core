// programs/roadhog.cpp
//
// Pulls together A*-based algorithms for
// Resource constrained and multi-objective shortest path problems
//
// @author: sahmadi
// @created: 01/11/23
// @updated: 24/06/24
//

#include "graph.h"
#include "input_parser.h"
#include <chrono>  // For high resolution timer

#include "search.h"
// #include "spherical_heuristic.h"
// #include "euclidean_heuristic.h"
// #include "haversine_heuristic.h"
#include "zero_heuristic.h"


// print debugging info during search
int verbose = 0;
// display program help on startup
int print_help = 0;
// reverse nodes
int rev_coor = 0;
// show path
int path = 0;
// for tie breaking
int tie = 0;
// for zero heuristic
int zero_heur = 0;
// number of repeats
int nruns = 1;

void
help()
{
	std::cerr
    << "==> manual <==\n"
    << "This program solves the point-to-point multi-objective/resource constrained shortest path problem (RCSP).\n"
    << "Input graphs are specified as xy-graphs. \n"
    << "XY format is similar to that used at the 9th DIMACS challenge.\n"
    << "Ref: http://www.diag.uniroma1.it//challenge9/download.shtml\n"
    << "The main differences are: \n"
    << "(i) a single file format consisting all objectives (cf. gr/co files) ; \n"
    << "(ii) node/arc ids are zero indexed (cf. 1-indexed) and; \n"
    << "Valid parameters:\n"
    << "\t--alg [ NWMOA, NWRCA, DCMOA]\n"
    << "\t\tNWMOA: A* for the multi-objective shortest path problem with negative weights (parallel search scheme). \n"
    << "\t\tNWRCA: A* for the resource constrained shortest path problem with negative weights. \n"
    << "\t\tDCMOA: A* for the doubly constrained multi-objective shortest path problem with negative weights. \n"
    << "\t--input [ graph_file.xy ] \n"
    << "\t--start [ start_ID ] (start vertex ID in 0-index) \n"
    << "\t--goal [ goal_ID ] (goal vertex ID in 0-index) \n"
    << "\t--constraint [1-100] (set resource constraint in % for RCSP algorithms) \n"
    << "\t--queue [bucket, hybrid] (priority queue type for the main search, default: bucket)\n"
    << "\t\tbucket: one-level bucket list with a linked list in buckets, works with integer costs\n"
    << "\t\thybrid: bucket list(higher level) and binary heap(lower-level), works with integer/non-integer costs\n"
    << "Valid flags:\n"
    << "\t--noheader (to supress header when printing the results)\n"
    << "\t--tie (to enable tie breaking on the second objective in the hybrid queue)\n"
    // << "\t--zero (to disable spherical heuristic, needed in random non-DIMACS instances)\n"
    << "\t--nruns #  (set number of repeats, default = 1)\n"
    << "\t--path  (to enable backtracking and print paths details)\n"
    << "\t--reverse  (to reverse source <-> target)\n";
}


template <class LABEL, class QUEUE, class HEURISTIC>
void
alg_manager_multiobj(graph *G, graph *G_rev, experiment exp)
{
    for(uint32_t run = 0; run < exp.nruns; run++)
    {
        if(exp.alg_name == "NWMOA" && G->Num_objectives == 3)
        {NWMOA_3C<LABEL, QUEUE, HEURISTIC>(G, G_rev, exp);}
        else if(exp.alg_name == "NWMOA")
        {NWMOA<LABEL, QUEUE, HEURISTIC>(G, G_rev, exp);}
        else if(exp.alg_name == "NWRCA")
        {NWRCA<LABEL, QUEUE, HEURISTIC>(G, G_rev, exp);}
        else if(exp.alg_name == "DCMOA")
        {DCMOA<LABEL, QUEUE, HEURISTIC>(G, G_rev, exp);}
        else
        {std::cerr << "invalid search algorithm "<< exp.alg_name <<std::endl; break;}
    }

}


void
execute_multiobj_search(graph *G, graph *G_rev, experiment exp)
{
    if (exp.queue_type == "heap" )
    {
        #ifdef PATH
        if (tie)
        {alg_manager_multiobj<search_label_pri, pqueue_label_heap_tie<search_label_pri>, zero_heuristic> (G, G_rev, exp);}
        else
        {alg_manager_multiobj<search_label_pri, pqueue_label_heap<search_label_pri>, zero_heuristic> (G, G_rev, exp);}
        #else
        if (tie)
        {alg_manager_multiobj<search_label_light_pri, pqueue_label_heap_tie<search_label_light_pri>, zero_heuristic> (G, G_rev, exp);}
        else
        {alg_manager_multiobj<search_label_light_pri, pqueue_label_heap<search_label_light_pri>, zero_heuristic> (G, G_rev, exp);}
        #endif
    }
    else if (exp.queue_type == "hybrid" )
    {
        #ifdef PATH
        if (tie)
        {alg_manager_multiobj<search_label_pri, pqueue_label_hybrid_cyclic_fixed_min_w_tie, zero_heuristic> (G, G_rev, exp);}
        else
        {alg_manager_multiobj<search_label_pri, pqueue_label_hybrid_cyclic_fixed_min_wo_tie, zero_heuristic> (G, G_rev, exp);}
        #else
        if (tie)
        {alg_manager_multiobj<search_label_light_pri, pqueue_label_light_hybrid_cyclic_fixed_min_w_tie, zero_heuristic> (G, G_rev, exp);}
        else
        {alg_manager_multiobj<search_label_light_pri, pqueue_label_light_hybrid_cyclic_fixed_min_wo_tie, zero_heuristic> (G, G_rev, exp);}
        #endif
    }
    else
    {
        #ifdef PATH
        if (tie)
        {alg_manager_multiobj<search_label, pqueue_label_bucket_fifo<search_label>, zero_heuristic> (G, G_rev, exp);}
        else
        {alg_manager_multiobj<search_label, pqueue_label_bucket_cyclic_fixed<search_label>, zero_heuristic> (G, G_rev, exp);}
        #else
        if (tie)
        {alg_manager_multiobj<search_label_light, pqueue_label_bucket_fifo<search_label_light>, zero_heuristic> (G, G_rev, exp);}
        else
        {alg_manager_multiobj<search_label_light, pqueue_label_bucket_cyclic_fixed<search_label_light>, zero_heuristic> (G, G_rev, exp);}
        #endif
    }
}

void
run_experiment(experiment& exp)
{
    if(exp.file_name == "")
    {
        std::cerr << "parameter is missing: --input [xy-graph file]\n";
        return;
    }

    if(exp.alg_name == "" || exp.start == SN_ID_MAX || exp.goal == SN_ID_MAX)
    {
        std::cerr << "At least one parameter is missing!\n";
        help();
        return;
    }

    bool is_rcsp = exp.alg_name == "NWRCA" ? true : false;
    if( is_rcsp && exp.constraint == UINT32_MAX )
    {
        std::cerr << "--constraint parameter is missing/invalid!\n";
        return;
    }
    else if(exp.constraint == UINT32_MAX )
    {exp.constraint = 100;}


    is_rcsp = exp.alg_name == "DCMOA" ? true : false;
    if( is_rcsp && exp.constraint == UINT32_MAX )
    {
        std::cerr << "--constraint parameter is missing/invalid!\n";
        return;
    }
    else if(exp.constraint == UINT32_MAX )
    {exp.constraint = 100;}

    if (exp.reverse)
    {   sn_id_t tmp = exp.start;
        exp.start = exp.goal;
        exp.goal = tmp;
    }

    if (!(exp.queue_type == "hybrid"))
    {exp.queue_type = "bucket";}

    graph *G = new graph();
    graph *G_rev = new graph();
    load_graph(G, G_rev, exp.file_name);

    if ( exp.start <= G->Num_vertices && exp.goal <= G->Num_vertices)
    {
        for(int run = 1; run <= nruns; run++)
        execute_multiobj_search(G, G_rev, exp);
    }
    else
    {
        if(exp.start - G->Vertex_offset > G->Num_vertices - 1){std::cerr<<"start_id does not exist.\n";}
        if(exp.goal  - G->Vertex_offset > G->Num_vertices - 1){std::cerr<<"goal_id does not exist.\n";}
    }

    delete G;
    delete G_rev;
}


int
main(int argc, char** argv)
{
	experiment exp;

    // parse arguments
    const char* const short_args = "a:i:s:g:p:r:q:c:n:";
    const option long_args[] =
	{
		{"alg",  required_argument, 0, 'a'},
		{"help", no_argument, &print_help, 1},
		{"noheader",  no_argument, &exp.suppress_header, 1},
		{"input",  required_argument, 0, 'i'},
        {"start",  required_argument, 0, 's'},
        {"goal",  required_argument, 0, 'g'},
        {"path",  no_argument, 0, 'p'},
        {"reverse",  no_argument, 0, 'r'},
        {"tie",  no_argument, &tie, 1},
        {"zero",  no_argument, &zero_heur, 1},
        {"queue",  required_argument, 0, 'q'},
        {"constraint",  required_argument, 0, 'c'},
        {"nruns",  required_argument, 0, 'n'},
	};

    parse_args(argc, argv, short_args, long_args, exp);

    if(argc == 1 || print_help)
    {
		help();
        exit(0);
    }

    run_experiment(exp);
}
