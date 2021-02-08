#pragma once
#include <stdint.h>
#include <vector_types.h>

struct Params
{
    uchar4*                image;
    uint32_t               image_width;
    uint32_t               image_height;
    int32_t                origin_x;
    int32_t                origin_y;
    OptixTraversableHandle handle;
};

struct RayGenData
{
    float3 cam_eye;
    float3 camera_u;
    float3 camera_v; 
    float3 camera_w;
    float  tmin ; 
    float  tmax ; 
};

struct MissData
{
    float r, g, b;
};

struct HitGroupData
{
    float* values ;
};

/**
HitGroupAnalyticCSGData
-------------------------

Maybe a templated type with integer template args 
can turn a concatenation of CSG shapes (from GParts) 
with variable numbers of prims, nodes, planes, transforms
into something of fixed size for the Sbt data.

Otherwise would need to access global buffers ?

What are the performance implications of that ?
**/

struct HitGroupAnalyticCSGData
{
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

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

#endif

