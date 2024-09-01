#ifndef MEMORY_H
#define MEMORY_H

// Memory tool, reading the real and virtual memory
//
// @author: sahmadi
//
// @updated: 05/06/2021

#include <string.h>
#include <stdint.h>
#include <iostream>
// #include <sstream>
#include <fstream>

class memory
{
public:
    void read_mem();
    uint64_t get_mem_real();
    uint64_t get_mem_real_peak();
    uint64_t get_mem_virt();
    uint64_t get_mem_virt_peak();

private:
    uint64_t real_mem;
    uint64_t real_mem_peak;
    uint64_t virt_mem;
    uint64_t virt_mem_peak;
};

#endif