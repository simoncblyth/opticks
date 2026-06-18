#pragma once

/**
Params.h
===========

* CPU side params instanciated in CSGOptiX::CSGOptiX and populated by CSGOptiX::init methods

**/

#include <optix.h>
#include <vector_types.h>

struct CSGNode ;
struct qat4 ;
struct quad4 ;
struct quad6 ;
struct qsim ;
struct sevent ;


struct __align__(16) Params // Force 16-byte structural alignment boundary
{
    // 1. 64-bit (8 byte) pointers and handles : need pairs of them to stay 16-byte aligned
    CSGNode*   node ;
    float4*    plan ;

    qat4*      tran ;
    qat4*      itra ;

    uchar4*    pixels ;
    float4*    isect ;

    quad4*     fphoton ;
    qsim*        sim ;

    sevent*      evt ;
    OptixTraversableHandle  handle ;

    unsigned long long  photon_slot_offset ;
    uint64_t   _pad0;


    // 2. 16-byte Vector types (float4, uint4) : these are inherently 16 byte aligned
    float4     center_extent ;
    float4     ZPROJ ;
    uint4      cegs ;
    float4     eye;
    float4     U ;
    float4     V ;
    float4     W ;
    float4     WNORM ;
    uint4      pidxyz ;

    // 3. 32-bit (4 byte) types (ints, floats, unsigned) : need groups of four to stay 16-byte aligned
    int32_t    raygenmode ;
    uint32_t   width;
    uint32_t   height;
    uint32_t   depth;

    uint32_t   cameratype ;
    int32_t    traceyflip ;
    int32_t    rendertype ;
    int32_t    origin_x;

    int32_t    origin_y;
    int        event_index ;
    float      tmin ;
    float      tmin0 ;

    float      PropagateRefineDistance ;
    float      tmax ;
    float      max_time ;
    unsigned   PropagateEpsilon0Mask ;

    unsigned   vizmask ;
    uint32_t   PropagateRefine ;
    uint32_t   _pad1;
    uint32_t   _pad2;

};


#ifndef __CUDACC__
#include <glm/glm.hpp>
#include <string>

struct Params_Helper
{
    Params* params   ;
    Params* d_params ;
    int     level ;

    Params_Helper(Params* params, int raygenmode, unsigned width, unsigned height, unsigned depth);
    void init();

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
    void setPIDXYZ(unsigned x, unsigned y, unsigned z, unsigned w);
    void set_photon_slot_offset(unsigned long long _photon_slot_offset);

};
#endif



