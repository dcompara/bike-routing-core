#include "memory.h"
// #include "labelpool.h"
// #include <stdio.h>
// #include <iostream>
// #include <fstream>

void 
memory::read_mem()
{
    // stores each word in status file
    char buffer[1024] = "";

    std::ifstream indata; // indata is like cin
    indata.open("/proc/self/status"); // opens the file
    if(!indata)
    { // file couldn't be opened
      std::cerr << "Error: memory file could not be opened" << std::endl;
      exit(1);
    }


    while ( !indata.eof() )
    { // keep reading until end-of-file
      indata >> buffer; // sets EOF flag if no value found
      if (strcmp(buffer, "VmRSS:") == 0)
        {
            indata >> real_mem;
        }
        if (strcmp(buffer, "VmHWM:") == 0)
        {
            indata >> real_mem_peak;
        }
        if (strcmp(buffer, "VmSize:") == 0)
        {
            indata >> virt_mem;
        }
        if (strcmp(buffer, "VmPeak:") == 0)
        {
            indata >> virt_mem_peak;
        }

    
    }
   indata.close();
}

uint64_t 
memory::get_mem_real(){return real_mem;}
uint64_t 
memory::get_mem_real_peak(){return real_mem_peak;}
uint64_t 
memory::get_mem_virt(){return virt_mem;}
uint64_t 
memory::get_mem_virt_peak(){return virt_mem_peak;}