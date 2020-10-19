#include <optix_world.h>

using namespace optix;
#include "PerRayData_pathtrace.h"

rtDeclareVariable(PerRayData_pathtrace, prd, rtPayload, );

rtDeclareVariable(float3, hit_pos, attribute hit_pos, );

RT_PROGRAM void closest_hit()
{
    rtPrintf("//closest_hit hit_pos(%f %f %f) \n", hit_pos.x, hit_pos.y, hit_pos.z ); 
    prd.hitID = hit_pos.x / 10; 
    //prd.hitID = 10u; 
};


