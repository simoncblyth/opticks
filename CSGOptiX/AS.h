#pragma once

#include <optix.h>

struct AS
{
    CUdeviceptr             d_buffer;  
    OptixTraversableHandle  handle ; 
};



