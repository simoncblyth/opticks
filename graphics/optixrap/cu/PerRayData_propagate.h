#pragma once

#include <optix.h>
#include <optix_math.h>

struct PerRayData_propagate
{
    float3 surface_normal ; 
    float distance_to_boundary ;
    int   boundary ; 
    int   sensor ; 
    uint4 identity ; 
    float cos_theta ;
};


