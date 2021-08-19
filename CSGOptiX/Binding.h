#pragma once
#include <stdint.h>
#include <vector_types.h>

struct RaygenData
{
    float placeholder ; 
};

struct MissData
{
    float r, g, b;
};

struct HitGroupData   // effectively Prim 
{
    int numNode ;   
    int nodeOffset ; 
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

