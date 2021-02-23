#pragma once

#include <vector>
#include <vector_types.h>
struct Params ; 

struct Frame
{
    Params* params ; 

    std::vector<uchar4> pixels ; 
    uchar4* d_pixels = nullptr ; 

    Frame(Params* params_);
 
    void init(); 
    void download(); 
    void writePPM(const char* path, bool yflip=true ); 
}; 


