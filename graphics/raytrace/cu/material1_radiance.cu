#include <optix.h>
#include "PerRayData_radiance.h"
#include "wavelength_lookup.h"

//geometric_normal is set by the closest hit intersection program 
rtDeclareVariable(float3, geometricNormal, attribute geometric_normal, );
rtDeclareVariable(unsigned int, nodeIndex, attribute node_index, );
rtDeclareVariable(float3, contrast_color, , );

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );



RT_PROGRAM void closest_hit_radiance()
{
  const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ; 

  const float cos_theta = dot(n,ray.direction);

  //prd_radiance.result = n*0.5f + 0.5f;

  prd_radiance.result = make_float3( 0.5f*(1.0f-cos_theta) );
  prd_radiance.node = nodeIndex ;


  /*
  float4 props = wavelength_lookup( NM_GREEN  , 0.5f ) ;
  float refractive_index  = props.x ; 
  float absorption_length = props.y ; 
  float scattering_length = props.z ; 
  float reemission_prob   = props.w ; 

  //prd_radiance.result.y = refractive_index - 1.0f  ; 
  //prd_radiance.result.y = reemission_prob  ; 

  */ 



  if(cos_theta > 0.0f )
  {
     prd_radiance.result.x = 1.0f ; 

     // make back faces red : given that the "light" is effectively coming from
     // the viewpoint this can probably only happen due to a geometry bug 
     //
     // * maybe surfaces too close to each other resulting in numerical precision flipping
     //   between alternate closest hit surfaces  
     // * flipped triangle winding order is not impossible
     // 
     // Little red is seen:
     //
     // * small red triangles at ends of some struts/ribs on top of AD
     // * when enter inside a PMT, see a concentric circle bullseye red/white pattern
     //   no problem is apparent for the external view of the PMT 
     // * from inside calibration assemblies quite a lot of speckly red/black
     // 
  }


  //prd_radiance.result = make_float3(0.f);

  //float u0 = curand_uniform(&prd_radiance.rng); 
  //float u1 = curand_uniform(&prd_radiance.rng); 
  //float u2 = curand_uniform(&prd_radiance.rng); 

  //prd_radiance.result = contrast_color ; 
  //prd_radiance.result.x = u0 ; 
  //prd_radiance.result.x = 

  //prd_radiance.result = make_float3( u0, u1, u2) ; 

 
  //prd_radiance.result = make_float3( u0, u1 , contrast_color.z) ; 




}


