#pragma once

#include <optix.h>

struct GAS
{
    OptixTraversableHandle gas_handle;
    CUdeviceptr            d_gas_output_buffer;


    GAS(); 
    void init(); 
    OptixTraversableHandle build(OptixBuildInput aabb_input); 

};


