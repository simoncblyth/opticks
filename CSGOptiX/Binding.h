#pragma once
/**
Binding.h
===========

See SBT::createHitgroup for population of CustomPrim and TriMesh

TODO:

1. remove placeholders
2. dont need 32 bits, so 16+16 globalPrimIdx_boundary
3. use globalPrimIdx_boundary in CustomPrim too - as srpn.h does not have space for boundary

**/



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


struct CustomPrim
{
    int numNode ;
    int nodeOffset ;
    int globalPrimIdx ;
};

struct TriMesh
{
    unsigned boundary ;
    uint3*  indice ;  // indices of triangle vertices and normal
    float3* vertex ;
    float3* normal ;
    int globalPrimIdx ;
};


struct HitGroupData
{
    union
    {
        CustomPrim prim ;
        TriMesh    mesh ;
    };
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

