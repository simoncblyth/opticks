
/**

Adapted from:: 

    optixrap/cu/csg_intersect_primitive.h
    optixrap/cu/csg_intersect_part.h

**/


#include <optix_world.h>
using namespace optix;

#include "math_constants.h"   // CUDART_ defines
#include "cu/quad.h"
#include "cu/Part.h"
#include "cu/bbox.h"
#include "cu/csg_intersect_primitive.h"


rtDeclareVariable(float4,  sphere, , );

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );



RT_PROGRAM void bounds(int /*primIdx*/, float result[6])
{
    quad q0 ; 
    q0.f = sphere ; 
    optix::Aabb* aabb = (optix::Aabb*)result;
    csg_bounds_sphere( q0, aabb, NULL ); 

    rtPrintf("//bounds result (%f %f %f) (%f %f %f) \n", result[0], result[1], result[2], result[3], result[4], result[5] );
}

RT_PROGRAM void intersect(int /*primIdx*/)
{
    quad q0 ; 
    q0.f = sphere ; 

    float t_min = 0.001f ;  
    float4 tt = make_float4(0.f,0.f,1.f, t_min);
    bool valid_intersect = csg_intersect_sphere(q0, t_min, tt, ray.origin, ray.direction ); 
    if(valid_intersect)
    {
        if(rtPotentialIntersection(tt.w))
        {
            shading_normal.x = geometric_normal.x = tt.x ;
            shading_normal.y = geometric_normal.y = tt.y ;
            shading_normal.z = geometric_normal.z = tt.z ;
            rtReportIntersection(0);
        }
    }
}

