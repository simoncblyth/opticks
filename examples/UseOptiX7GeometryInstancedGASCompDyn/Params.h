#pragma once

#include <optix.h>
#include <vector_types.h>


struct Params
{
    uchar4*    pixels ;
    float4*    isect ;

    uint32_t   width;
    uint32_t   height;
    uint32_t   depth;
    uint32_t   cameratype ; 

    int32_t    origin_x;
    int32_t    origin_y;

    float3     eye;
    float3     U ;
    float3     V ; 
    float3     W ;
    float      tmin ; 
    float      tmax ; 

    OptixTraversableHandle handle;

};

