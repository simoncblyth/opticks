#include "switches.h"

#include <optix.h>
#include <optix_math.h>
#include "PerRayData_radiance.h"

//geometric_normal is set by the closest hit intersection program 
rtDeclareVariable(float3, geometricNormal, attribute geometric_normal, );
rtDeclareVariable(uint4,  instanceIdentity, attribute instance_identity, );

rtDeclareVariable(float3, contrast_color, , );

rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t,            rtIntersectionDistance, );

rtDeclareVariable(unsigned int,  touch_mode, , );
rtDeclareVariable(float4,        ZProj, , );
rtDeclareVariable(float3,        front, , );
rtDeclareVariable(unsigned int,  parallel, , );


RT_PROGRAM void closest_hit_radiance()
{
    const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ; 
    const float cos_theta = dot(n,ray.direction);

    float intensity = 0.5f*(1.0f-cos_theta) ;  // lambertian 

    float zHit_eye = -t*dot(front, ray.direction) ;   // intersect z coordinate (eye frame), always -ve 
    float zHit_ndc = parallel == 0 ? -ZProj.z - ZProj.w/zHit_eye : ZProj.z*zHit_eye + ZProj.w ;  // should be in range -1:1 for visibles
    float zHit_clip = 0.5f*zHit_ndc + 0.5f ;   // 0:1 for visibles

    //rtPrintf("closest_hit_radiance t %10.4f zHit_eye %10.4f  ZProj.z %10.4f ZProj.w %10.4f zHit_ndc %10.4f zHit_clip %10.4f \n", t, zHit_eye, ZProj.z, ZProj.w , zHit_ndc, zHit_clip );

    prd.result = make_float4(intensity, intensity, intensity, zHit_clip ); // hijack alpha for the depth 

#ifdef BOOLEAN_DEBUG
     switch(instanceIdentity.x)
     {
        case 1: prd.result.x = 1.f ; break ;
        case 2: prd.result.y = 1.f ; break ;
        case 3: prd.result.z = 1.f ; break ;
    }
#endif    

    prd.flag   = instanceIdentity.y ;   //  hijacked to become the hemi-pmt intersection code
}



