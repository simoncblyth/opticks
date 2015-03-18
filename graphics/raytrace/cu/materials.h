#include <optix.h>
#include <optix_math.h>
#include "commonStructs.h"

#define NM_BLUE   475.f
#define NM_GREEN  510.f
#define NM_RED    650.f


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

  float4 texlookup_b ;
  float4 texlookup_g ;
  float4 texlookup_r ;

};





