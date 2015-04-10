#ifndef PERRAYDATA_HYBRID_H
#define PERRAYDATA_HYBRID_H

#include <optix.h>
#include <optix_math.h>
#include "commonStructs.h"

//#include "RayTraceConfigInc.h"
#include <curand_kernel.h>

struct PerRayData_hybrid
{
  float3 result;
  float  importance;
  int    depth;

  // hmm: keeping photon properties here will 
  // entail copying forward as optix::Ray are created
  // at each bounce

  unsigned int history ;
  float4 position_time ;         // unlike ray.origin,  position_time is propagated forward  
  float4 direction_wavelength ;  // hmm same as ray.direction ?  
  float4 polarization_weight ;  

  curandState rng;
};


#endif
