#ifndef PERRAYDATA_RADIANCE_H
#define PERRAYDATA_RADIANCE_H

#include <optix.h>
#include <optix_math.h>
#include "commonStructs.h"

//#include "RayTraceConfigInc.h"
#include <curand_kernel.h>

struct PerRayData_radiance
{
  float3 result;
  float  importance;
  int    depth;
  unsigned int node ; 
  curandState rng;
};


#endif
