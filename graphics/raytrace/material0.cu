// /Users/blyth/env/cuda/optix/OptiX_301/materials/material0.cu

#include "materials.h"

// Variables are set host-side
rtDeclareVariable(float3, Kd, , );

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );

//
// Returns a solid color as the shading result 
// 
RT_PROGRAM void closest_hit_radiance()
{
  prd_radiance.result = Kd; 
}
