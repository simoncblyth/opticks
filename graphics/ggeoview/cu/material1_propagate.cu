#include <optix.h>
#include "PerRayData_propagate.h"
#include "wavelength_lookup.h"

//geometric_normal is set by the closest hit intersection program 
rtDeclareVariable(float3, geometricNormal, attribute geometric_normal, );
rtDeclareVariable(float3, intersectionPosition, attribute intersection_position, );
rtDeclareVariable(unsigned int, nodeIndex, attribute node_index, );
rtDeclareVariable(unsigned int, substanceIndex, attribute substance_index, );

rtDeclareVariable(PerRayData_propagate, prd, rtPayload, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );


RT_PROGRAM void closest_hit_propagate()
{
     const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ; 

     const float cos_theta = dot(n,ray.direction);

     prd.intersection = intersectionPosition ; 

     prd.depth = nodeIndex ;

}


