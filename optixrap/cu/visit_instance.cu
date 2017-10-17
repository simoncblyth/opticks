
#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
//rtDeclareVariable(float4, instance_position, , );
rtDeclareVariable(float, instance_size, , );

RT_PROGRAM void visit_instance()
{
    /*
        0  1  2  3
        4  5  6  7
        8  9 10 11
       12 13 14 15 
    */

    float matrix[16] ;
 
    rtGetTransform( RT_OBJECT_TO_WORLD , matrix ) ;

    const float3 ipos = make_float3( matrix[3], matrix[7], matrix[11] );  // 12 13 14 was (0,0,0)

    const float3 orig = rtTransformPoint( RT_OBJECT_TO_WORLD, ray.origin );

    const float3 offset = orig - ipos  ;  

    float distance = length( offset ) ; 

    unsigned level = distance < instance_size ? 0u : 1u ;  

    //if(level == 0u)
    rtPrintf("visit_instance: level %d instance_size %10.3f distance %10.3f  orig (%10.3f %10.3f %10.3f) jpos (%10.3f %10.3f %10.3f)  \n", 
          level, 
          instance_size,
          distance,
          orig.x, orig.y, orig.z,
          ipos.x, ipos.y, ipos.z
       ); 

    rtIntersectChild( level );
}


