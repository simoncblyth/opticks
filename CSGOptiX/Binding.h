#pragma once
#include <stdint.h>
#include <vector_types.h>

/**
TODO: try removing the placeholders

**/

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
};

struct TriMesh
{
    unsigned boundary ;  
    uint3*  indice ;  // indices of triangle vertices and normal 
    float3* vertex ; 
    float3* normal ; 
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

