
#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float4, instance_position, , );

RT_PROGRAM void visit_instance()
{
    const float3 origin = rtTransformPoint( RT_OBJECT_TO_WORLD, ray.origin );

    const float3 offset = origin - make_float3( instance_position ) ;  

    float distance = length( offset ) ; 

    const float size = instance_position.w ; 

    unsigned index = (unsigned)( distance < size ); // 

    rtPrintf("visit_instance: index %d size %10.3f distance %10.3f    origin (%10.3f %10.3f %10.3f) instance_position (%10.3f %10.3f %10.3f %10.3f)  \n", 
          index, 
          size,
          distance,
          origin.x, origin.y, origin.z, 
          instance_position.x, instance_position.y, instance_position.z, instance_position.w ); 

    rtIntersectChild( index );
}


