#pragma once

#include <optix.h>
#include <optix_math.h>

struct PerRayData_propagate
{
    float distance_to_boundary ;
    int   boundary ; 
    float cos_theta ;
};


