#ifndef CONSTANTS_H
#define CONSTANTS_H

// constants.h
//
// @author: sahmadi
// @updated: 06/05/2021
//

#include <cfloat>
#include <cmath>
#include <climits>
#include <stdint.h>
#include <vector>
#include <string>
#include <stdint.h>
#include <array>
#include "arraylist.h"
#include "linkedlist.h"

typedef uint32_t uint;

// address space for state identifiers
typedef uint32_t sn_id_t;
static const sn_id_t SN_ID_MAX = UINT32_MAX;

// For non-negative integer costs
// typedef uint32_t cost_t;
// static const cost_t COST_MAX = UINT32_MAX;
// static const cost_t COST_MIN = 0;

// For non-negative integer costs
typedef int32_t cost_t;
static const cost_t COST_MAX = INT32_MAX;
static const cost_t COST_MIN = INT32_MIN;

// For floating point costs
// typedef double cost_t;
// static const cost_t COST_MAX = DBL_MAX;
// static const cost_t COST_MIN = DBL_MIN;

// This parameter sets the dimension of the cost vectors.
typedef uint8_t dim_t;
const dim_t DIM = 2;

typedef uint8_t vertex_deg_t ;
const vertex_deg_t DEG_MAX = UINT8_MAX;

typedef cost_t edge_cost_t;
const cost_t EDGE_COST_MAX = INT32_MAX;

typedef uint16_t path_arr_size;
static const path_arr_size PATH_ARR_SIZE_MAX = UINT16_MAX;
typedef std::vector<std::pair<vertex_deg_t, path_arr_size>> Parent_list;
// typedef arraylist<std::pair<vertex_deg_t, path_arr_size>> Parent_list;
typedef std::pair<int32_t, int32_t> coord_pair;

struct experiment
{
    sn_id_t start = SN_ID_MAX;
    sn_id_t goal = SN_ID_MAX;
    uint32_t constraint = UINT32_MAX;
    uint32_t path = 0;
    uint32_t reverse = 0;
    uint32_t nruns = 1;
    int suppress_header = 0;
    std::string alg_name;
    std::string file_name;
    std::string map_name;
    std::string queue_type;
    static dim_t num_objs;
};

struct edge
{
    sn_id_t tail;
    vertex_deg_t tail_incoming;
    std::array<edge_cost_t, DIM> costs;
};
typedef struct edge edge_t;
typedef std::vector<std::vector<edge_t>> edge_data_t;
#endif
