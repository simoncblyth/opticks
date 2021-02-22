#pragma once
#include <stdint.h>
#include <vector_types.h>

#define SBT_VIEW 1
#ifdef SBT_VIEW
struct RaygenData
{
    float3 eye;
    float3 U;
    float3 V; 
    float3 W;
    float  tmin ; 
    float  tmax ; 
};
#else
struct RaygenData
{
    float placeholder ; 
};

#endif


struct MissData
{
    float r, g, b;
};

struct HitGroupData
{
    float* values ;
};


#if defined(__CUDACC__) || defined(__CUDABE__)
#else

#include <optix_types.h>

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RaygenData>     Raygen ;
typedef SbtRecord<MissData>       Miss ;
typedef SbtRecord<HitGroupData>   HitGroup ;

#endif

