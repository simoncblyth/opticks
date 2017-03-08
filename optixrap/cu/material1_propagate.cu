#include <optix.h>
#include "PerRayData_propagate.h"
#include "wavelength_lookup.h"

//attributes set by TriangleMesh.cu:mesh_intersect 

rtDeclareVariable(float3,  geometricNormal, attribute geometric_normal, );
rtDeclareVariable(uint4,  instanceIdentity, attribute instance_identity, );

rtDeclareVariable(PerRayData_propagate, prd, rtPayload, );
rtDeclareVariable(optix::Ray,           ray, rtCurrentRay, );
rtDeclareVariable(float,                  t, rtIntersectionDistance, );


RT_PROGRAM void closest_hit_propagate()
{
     const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ; 

     float cos_theta = dot(n,ray.direction); 

     prd.cos_theta = cos_theta ;

     prd.distance_to_boundary = t ;   // huh: there is an standard attrib for this

     unsigned int boundaryIndex = instanceIdentity.z ; 

     prd.boundary = cos_theta < 0.f ? -(boundaryIndex + 1) : boundaryIndex + 1 ;   

     prd.identity = instanceIdentity ; 

     prd.surface_normal = cos_theta > 0.f ? -n : n ;   
 
}

// prd.boundary 
//    * 1-based index with cos_theta signing, 0 means miss
//    * sign identifies which of inner/outer-material is material1/material2 
//    * by virtue of zero initialization, a miss leaves prd.boundary at zero
//
//  cos_theta > 0.f
//        outward going photons, with p.direction in same hemi as the geometry normal
//
//  cos_theta < 0.f  
//        inward going photons, with p.direction in opposite hemi to geometry normal
//
// surface_normal oriented to point from material2 back into material1 
//

