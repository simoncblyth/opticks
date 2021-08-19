#pragma once

#include <optix.h>
// TODO: avoid need for optix.h just for OptixTraversableHandle which is "unsigned long long" typedef

#include <vector_types.h>

#ifndef __CUDACC__
#include <glm/glm.hpp>
#endif

struct CSGNode ; 
struct qat4 ; 
struct quad4 ; 
struct quad6 ; 


struct Params
{
    // render/simulation switch 
    int32_t    raygenmode ;

    // geometry : from Foundry 
    CSGNode*   node ; 
    float4*    plan ; 
    qat4*      tran ; 
    qat4*      itra ; 

#if OPTIX_VERSION < 70000
    void*                   handle ;  
#else
    OptixTraversableHandle  handle ; 
#endif

    // frame rendering 
    uchar4*    pixels ;
    float4*    isect ;

    // render control 
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


    // simulation 
    uint32_t   num_photons ; 
    uint32_t   num_gensteps ; 
    quad4*     photons ; 
    quad6*     gensteps ; 



#ifndef __CUDACC__
    Params(int raygenmode, unsigned width, unsigned height, unsigned depth); 
    void setView(const glm::vec4& eye_, const glm::vec4& U_, const glm::vec4& V_, const glm::vec4& W_ );
    void setView(const glm::vec3& eye_, const glm::vec3& U_, const glm::vec3& V_, const glm::vec3& W_ );
    void setCamera(float tmin_, float tmax_, unsigned cameratype_ ) ;
    void setRaygenMode(int raygenmode_ );
    void setSize(unsigned width_, unsigned height_, unsigned depth_ );
#endif

};

