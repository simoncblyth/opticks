#include <optix.h>
#include <optix_math.h>
#include "commonStructs.h"

struct PerRayData_radiance
{
  float3 result;
  float  importance;  // This is ignored in this sample.  See phong.h for use.
  int    depth;
};


struct PerRayData_shadow
{
  float3 attenuation;
};


struct PerRayData_touch
{
  float3 result;
  int    depth;
  unsigned int  node;
  float4 texlookup_l ;
  float4 texlookup_m ;
  float4 texlookup_r ;

};





