#pragma once


#include <stdint.h>



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
};


struct MissData
{
    float r, g, b;
};


struct HitGroupData
{
    float radius;
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






