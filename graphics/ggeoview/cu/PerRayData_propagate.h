#pragma once

#include <optix.h>
#include <optix_math.h>
#include "commonStructs.h"

#include <curand_kernel.h>

struct PerRayData_propagate
{
  float3 intersection ;
  int    depth;
  curandState rng;
};


