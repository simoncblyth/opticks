#include "materials.h"

// shadingNormal is set by the closest hit intersection program 
rtDeclareVariable(float3, shading_normal, attribute shadingNormal, );

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );

//
// Returns a solid color as the shading result 
// 
RT_PROGRAM void closest_hit_radiance()
{
  //prd_radiance.result = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal))*0.5f + 0.5f;
  prd_radiance.result = make_float3(0.f);
}
