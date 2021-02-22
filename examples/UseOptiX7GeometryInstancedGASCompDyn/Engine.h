#pragma once

#include <vector>
#include <vector_types.h>

struct Geo ; 
struct PIP ; 
struct SBT ; 
struct AS ; 
struct Params ; 

struct Engine
{
    Params* params ; 
    PIP*    pip ;   
    SBT*    sbt ; 

    std::vector<uchar4> host_pixels ; 
    uchar4* d_pixels = nullptr ; 

    Engine(const char* ptx_path_, Params* params); 
    void init(); 

    void setGeo(const Geo* geo);
    void allocOutputBuffer(); 
    void launch(); 
    void download(); 
    void writePPM(const char* path, bool yflip=true ); 
}; 


