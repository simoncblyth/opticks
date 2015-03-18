#include <optix.h>
#include "PerRayData_radiance.h"

//geometric_normal is set by the closest hit intersection program 
rtDeclareVariable(float3, geometricNormal, attribute geometric_normal, );
rtDeclareVariable(float3, contrast_color, , );

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );


RT_PROGRAM void closest_hit_radiance()
{
  float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ; 

  float side = dot(n,-ray.direction);

  prd_radiance.result = n*0.5f + 0.5f;

  //prd_radiance.result = make_float3(0.f);

  //float u0 = curand_uniform(&prd_radiance.rng); 
  //float u1 = curand_uniform(&prd_radiance.rng); 
  //float u2 = curand_uniform(&prd_radiance.rng); 

  //prd_radiance.result = contrast_color ; 
  //prd_radiance.result.x = u0 ; 
  //prd_radiance.result.x = side ; 

  //prd_radiance.result = make_float3( u0, u1, u2) ; 
  prd_radiance.result = make_float3( side ) ; 
  //prd_radiance.result = make_float3( u0, u1 , contrast_color.z) ; 

}


