#pragma once

#include <optix.h>
#include <optix_math.h>
#include "commonStructs.h"

#include <curand_kernel.h>

struct PerRayData_propagate
{
  //float3 result;
  //float  importance;
  int    depth;
  //unsigned int node ; 
  curandState rng;
};


