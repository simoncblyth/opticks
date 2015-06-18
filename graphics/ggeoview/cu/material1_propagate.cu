#include <optix.h>
#include "PerRayData_propagate.h"
#include "wavelength_lookup.h"


//  see TriangleMesh.cu:mesh_intersect for where these attributes come from

rtDeclareVariable(float3,       geometricNormal, attribute geometric_normal, );

rtDeclareVariable(unsigned int, nodeIndex,       attribute node_index, );
rtDeclareVariable(unsigned int, boundaryIndex,   attribute boundary_index, );
rtDeclareVariable(unsigned int, sensorIndex,     attribute sensor_index, );

rtDeclareVariable(PerRayData_propagate, prd, rtPayload, );
rtDeclareVariable(optix::Ray,           ray, rtCurrentRay, );
rtDeclareVariable(float,                  t, rtIntersectionDistance, );


/*

PDF:

    A common use case of variable transformation occurs when interpreting
    attributes passed from the intersection program to the closest hit program.
    Intersection programs often produce attributes, such as normal vectors, in
    object space. Should a closest hit program wish to consume that attribute, it
    often must transform the attribute from object space to world space:


Debugging confirms that as not currently using any transforms within OptiX 
there is no need to do the transform.

*/


RT_PROGRAM void closest_hit_propagate()
{
    // contrast to /usr/local/env/chroma_env/src/chroma/chroma/cuda/photon.h:fill_state

     const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ; 

     //rtPrintf("closest_hit_propagate  n %10.4f %10.4f %10.4f     %10.4f \n", n.x, n.y, n.z, cos_theta );

     //
     // geometry normals expected to canonically be "outwards" (think cylinder) 
     // this is controlled by triangle index winding order  
     //
     // canonical photon direction is "outwards" 
     // so no sign flips to make cosTheta = +1 correspond to 
     // this canonical situation
     //
     // however to work with typical snells law conventions
     // need to point the normal in reflected direction ?
     //  
     //
     //  cos_theta > 0.f
     //          ray pointing in same hemisphere as outward normal
     //   

     float cos_theta = dot(n,ray.direction);

     prd.cos_theta = cos_theta ;

     prd.distance_to_boundary = t ; 

     // boundary sign identifies which of inner/outer-material is material1/material2 
     //
     //
     //  cos_theta > 0.f
     //        outward going photons, with p.direction in same hemi as the geometry normal
     //
     //  cos_theta < 0.f  
     //        inward going photons, with p.direction in opposite hemi to geometry normal
     //

     prd.boundary = cos_theta < 0.f ? -(boundaryIndex + 1) : boundaryIndex + 1 ;   // 1-based with cos_theta signing, 0 means miss

     prd.sensor = sensorIndex ; 

     prd.surface_normal = cos_theta > 0.f ? -n : n ;   
     //
     // orient normal to point from material2 back into material1 
     //
     // CAUTION: danger of doing a double flip, but need to arrange surface_normal according 
     //          to convention of fresnel derivation
     // 
     // Boundary sign arranges in fill_state that the raw (inner/outer material) 
     // are selected into: 
     //
     // * material1 source material
     // * material2 destination material
     //
 
}


