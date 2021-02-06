#pragma once

#include <vector>
#include <optix.h>

struct GAS
{
    CUdeviceptr             d_buffer;   // do I need to hang on to the buffer ? perhaps for cleanup ?
    OptixTraversableHandle  handle ; 

    static GAS Build(const std::vector<float>& bb ); // must contain a multiple of 6 min/max bbox floats  
};


