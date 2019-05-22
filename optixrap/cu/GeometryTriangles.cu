#include <optix.h>
#include <optixu/optixu_math_namespace.h>

#if OPTIX_VERSION >= 60000

using namespace optix;

// inputs from OGeo
rtBuffer<uint3>   indexBuffer; 
rtBuffer<float3> vertexBuffer;     
rtBuffer<uint4>  identityBuffer; 
rtDeclareVariable(unsigned int, instance_index,  ,);
rtDeclareVariable(unsigned int, primitive_count, ,);

// attribute variables communicating from intersection program to closest hit program
// (must be set between rtPotentialIntersection and rtReportIntersection)
//
// hmm but what about GeometryTriangles triangle_attributes ?

rtDeclareVariable(uint4, instanceIdentity,   attribute instance_identity,);
rtDeclareVariable(float3, geometricNormal, attribute geometric_normal, ); 


RT_PROGRAM void triangle_attributes()
{
    const int primIdx = rtGetPrimitiveIndex() ;
    const uint3  index  = indexBuffer[primIdx];

    const float3 p0    = vertexBuffer[index.x];
    const float3 p1    = vertexBuffer[index.y];
    const float3 p2    = vertexBuffer[index.z];
    const float3 norm    = optix::normalize(optix::cross( p1 - p0, p2 - p0 ));
    const uint4 identity = identityBuffer[instance_index*primitive_count+primIdx] ;  // index just primIdx for non-instanced

    geometricNormal = norm ;
    instanceIdentity = identity ;
}


#endif


