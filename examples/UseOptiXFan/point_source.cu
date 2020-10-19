#include <optix_world.h>
#include "random.h"  // OptiX random header file in SDK/cuda/random.h

using namespace optix;

#define PI 3.1415926f

#include "PerRayData_pathtrace.h"

rtDeclareVariable(float3, source_pos, , );  // position of source
rtDeclareVariable(rtObject, top_object, , );  // group object
rtDeclareVariable(uint, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint, launch_dim, rtLaunchDim, );
rtBuffer<unsigned, 1> output_id;  // record the id of dom that photon hit, 0 if no hit

rtDeclareVariable(PerRayData_pathtrace, prd, rtPayload, );


RT_PROGRAM void point_source()
{
    float3 ray_origin = source_pos;
    PerRayData_pathtrace prd;
    prd.hitID = 0u ;
    prd.seed = tea<4>(launch_index, 0u);
    float cos_th = 2.f * rnd(prd.seed) - 1.f;
    float sin_th = sqrt(1.f - cos_th*cos_th);
    float phi = 2.f * PI * rnd(prd.seed) ; 
    float cos_ph = cos(phi);
    float sin_ph = sin(phi); 
    float3 ray_direction = make_float3(cos_th*cos_ph, cos_th*sin_ph, sin_th);
    rtPrintf("//point_source  ray_direction (%f %f %f) ray_origin (%f %f %f)  \n", 
           ray_direction.x, ray_direction.y, ray_direction.z,
           ray_origin.x, ray_origin.y, ray_origin.z
           );  

    Ray ray = make_Ray(ray_origin, ray_direction, 0u, 0.01f, RT_DEFAULT_MAX);
    rtTrace(top_object, ray, prd);
    output_id[launch_index] = prd.hitID;
}


RT_PROGRAM void exception()
{
    rtPrintExceptionDetails();
}

RT_PROGRAM void miss()
{
    prd.hitID = 42u ;
}

