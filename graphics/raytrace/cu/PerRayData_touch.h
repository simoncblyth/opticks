#ifndef PERRAYDATA_TOUCH_H
#define PERRAYDATA_TOUCH_H

#include <optix.h>
#include <optix_math.h>

struct PerRayData_touch
{
  float3 result;
  int    depth;
  unsigned int  node;

  float4 texlookup_b ;
  float4 texlookup_g ;
  float4 texlookup_r ;

};

#endif

