
#include <optix.h>
#include "PerRayData_angular_propagate.h"
#include "reemission_lookup.h"
#include "source_lookup.h"

rtDeclareVariable(float3,  geometricNormal, attribute geometric_normal, );
rtDeclareVariable(uint4,  instanceIdentity, attribute instance_identity, );

rtDeclareVariable(PerRayData_angular_propagate, prd, rtPayload, );
rtDeclareVariable(optix::Ray,                   ray, rtCurrentRay, );
rtDeclareVariable(float,                          t, rtIntersectionDistance, );


RT_PROGRAM void closest_hit_angular_propagate()
{
     const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ; 
     float cos_theta = dot(n,ray.direction); 

     prd.distance_to_boundary = t ;   // standard semantic attrib for this not available in raygen, so must pass it

     unsigned boundaryIndex = ( instanceIdentity.z & 0xffff ) ; 
     prd.boundary = cos_theta < 0.f ? -(boundaryIndex + 1) : boundaryIndex + 1 ;   
     prd.identity = instanceIdentity ; 
     prd.surface_normal = cos_theta > 0.f ? -n : n ;   

     // for angular efficiency 
     const float3 isect = ray.origin + t*ray.direction ; 
     //const float3 local_point = rtTransformPoint( RT_WORLD_TO_OBJECT, isect ); 
     const float3 local_point_norm = normalize(rtTransformPoint( RT_WORLD_TO_OBJECT, isect )); 

     const float f_theta = acos( local_point_norm.z )/M_PIf;                             // polar 0->pi ->  0->1
     const float f_phi_ = atan2( local_point_norm.y, local_point_norm.x )/(2.f*M_PIf) ;  // azimuthal 0->2pi ->  0->1
     const float f_phi = f_phi_ > 0.f ? f_phi_ : f_phi_ + 1.f ;  //  
     prd.f_theta = f_theta ; 
     prd.f_phi = f_phi ; 
}



