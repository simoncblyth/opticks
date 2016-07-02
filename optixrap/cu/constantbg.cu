#include <optix_world.h>

rtDeclareVariable(float4, bg_color, , );

struct PerRayData_radiance
{
  float4 result;
  float importance;
  int depth;
};

rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );

RT_PROGRAM void miss()
{
  prd_radiance.result = bg_color;
}
