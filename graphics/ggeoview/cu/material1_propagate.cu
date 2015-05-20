#include <optix.h>
#include "PerRayData_propagate.h"
#include "wavelength_lookup.h"


//  see TriangleMesh.cu:mesh_intersect for where these attributes come from

rtDeclareVariable(float3,       geometricNormal, attribute geometric_normal, );
rtDeclareVariable(unsigned int, nodeIndex,       attribute node_index, );
rtDeclareVariable(unsigned int, substanceIndex,  attribute substance_index, );

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

     prd.cos_theta = dot(n,-ray.direction);

     prd.distance_to_boundary = t ; 

     prd.boundary = substanceIndex ;   

     // hmm maybe use 1-based substance index, signed according to cos_theta inner-to-outer  

}


