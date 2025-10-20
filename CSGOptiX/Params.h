#pragma once

/**
Params.h
===========

* CPU side params instanciated in CSGOptiX::CSGOptiX and populated by CSGOptiX::init methods

**/

#include <optix.h>
// TODO: avoid need for optix.h just for OptixTraversableHandle which is "unsigned long long" typedef

#include <vector_types.h>

#ifndef __CUDACC__
#include <glm/glm.hpp>
#include <string>
#endif

struct CSGNode ;
struct qat4 ;
struct quad4 ;
struct quad6 ;
struct qsim ;
struct sevent ;


struct Params
{
    // render/simtrace/simulate switch
    int32_t    raygenmode ;

    // geometry : from Foundry
    // notice no CSGPrim here as that is just a sequence of CSGNode handled by
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
    quad4*     fphoton ;

    // render control
    uint32_t   width;
    uint32_t   height;
    uint32_t   depth;
    uint32_t   cameratype ;

    int32_t    traceyflip ;
    int32_t    rendertype ;
    int32_t    origin_x;
    int32_t    origin_y;

    float3     eye;
    float3     U ;
    float3     V ;
    float3     W ;
    float3     WNORM ;
    float      tmin ;

    float      tmin0 ;
    unsigned   PropagateEpsilon0Mask ;
    float      PropagateRefineDistance ;
    bool       PropagateRefine ;

    float      tmax ;
    float4     ZPROJ ;

    unsigned   vizmask ;

    float4     center_extent ;
    uint4      cegs ;

    // simulation
    qsim*        sim ;
    sevent*      evt ;         // HMM: inside sim too ?
    int  event_index ;
    unsigned long long  photon_slot_offset ;   // for multi-launch to match single-launch
    float max_time ;           // ns


    // debug dumping : set from PIDXYZ envvar by CSGOptiX::initPIDXYZ default -1:-1:-1
    uint3     pidxyz ;


#ifndef __CUDACC__
    static Params* d_param ;

    Params(int raygenmode, unsigned width, unsigned height, unsigned depth);
    void device_alloc();
    void upload();

    std::string desc() const ;
    std::string detail() const ;

    void setView(const glm::vec3& eye_,
                 const glm::vec3& U_,
                 const glm::vec3& V_,
                 const glm::vec3& W_,
                 const glm::vec3& WNORM_ );

    void setCamera(float tmin_, float tmax_, unsigned cameratype_, int traceyflip_, int rendertype_, const glm::vec4& ZPROJ_ ) ;
    void setRaygenMode(int raygenmode_ );
    void setSize(unsigned width_, unsigned height_, unsigned depth_ );
    void setVizmask(unsigned vizmask_);

    void setCenterExtent(float x, float y, float z, float w);  // used for "simulation" planar rendering
    void setPIDXYZ(unsigned x, unsigned y, unsigned z);
    void set_photon_slot_offset(unsigned long long _photon_slot_offset);

#endif

};

