#include "PerRayData.h"

rtDeclareVariable(float3, Kd, , );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );

RT_PROGRAM void closest_hit_radiance()
{
    prd_radiance.result = Kd; 
}


