#include "materials.h"

// shadingNormal is set by the closest hit intersection program 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(unsigned int, node_index, attribute node_index, );

rtDeclareVariable(float3, contrast_color, , );


rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_touch, prd_touch, rtPayload, );

//
// 
RT_PROGRAM void closest_hit_radiance()
{
  //prd_radiance.result = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal))*0.5f + 0.5f;
  //prd_radiance.result = make_float3(0.f);
  prd_radiance.result = contrast_color ; 
}


RT_PROGRAM void closest_hit_touch()
{
  prd_touch.result = contrast_color ; 
  prd_touch.node = node_index ; 
}


