#pragma once
#include <optixu/optixu_vector_types.h>

typedef struct struct_BasicLight
{
#if defined(__cplusplus)
  typedef optix::float3 float3;
#endif
  float3 pos;
  float3 color;
  int    casts_shadow; 
  int    padding;      // make this structure 32 bytes -- powers of two are your friend!
} BasicLight;

struct TriangleLight
{
#if defined(__cplusplus)
  typedef optix::float3 float3;
#endif
  float3 v1, v2, v3;
  float3 normal;
  float3 emission;
};

