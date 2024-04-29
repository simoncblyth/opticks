#pragma once
/**
SOPTIX_BuildInput_CPA
======================

As SCSGPrimSpec are passed around by value do not assume long lived pointer addresses
back in the  SCSGPrimSpec.

Because the buildInputCPA.aabbBuffers requires a host array pointer to device pointers 
it is necessary to keep instances of this struct alive until at least Acceleration structure
creation. 

**/

#include "SCSGPrimSpec.h"
#include "SCU.h"
#include "SOPTIX_BuildInput.h"

struct SOPTIX_BuildInput_CPA : public SOPTIX_BuildInput
{
    static constexpr const char* NAME = "BuildInputCustomPrimitiveArray" ; 
    CUdeviceptr d_aabb ;    
    CUdeviceptr d_sbt_index ;
    unsigned*   flags ; 

    SOPTIX_BuildInput_CPA( const SCSGPrimSpec& ps ); 
};

 
inline SOPTIX_BuildInput_CPA::SOPTIX_BuildInput_CPA( const SCSGPrimSpec& ps )
    :
    SOPTIX_BuildInput(NAME),
    d_aabb(SCU::DevicePointerCast<float>( ps.aabb )),
    d_sbt_index(SCU::DevicePointerCast<unsigned>( ps.sbtIndexOffset )),
    flags(new unsigned[ps.num_prim])
{
    for(unsigned i=0 ; i < ps.num_prim ; i++) flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT ; 

    buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    OptixBuildInputCustomPrimitiveArray& buildInputCPA = buildInput.customPrimitiveArray ;   

    buildInputCPA.aabbBuffers = &d_aabb ;   
    buildInputCPA.numPrimitives = ps.num_prim  ;   
    buildInputCPA.strideInBytes = ps.stride_in_bytes ;
    buildInputCPA.flags = flags;                                     // flags per sbt record
    buildInputCPA.numSbtRecords = ps.num_prim ;                      // number of sbt records available to sbt index offset override. 
    buildInputCPA.sbtIndexOffsetBuffer  = d_sbt_index ;              // Device pointer to per-primitive local sbt index offset buffer, Every entry must be in range [0,numSbtRecords-1]
    buildInputCPA.sbtIndexOffsetSizeInBytes  = sizeof(unsigned);     // Size of type of the sbt index offset. Needs to be 0,1,2 or 4    
    buildInputCPA.sbtIndexOffsetStrideInBytes = ps.stride_in_bytes ; // Stride between the index offsets. If set to zero, the offsets are assumed to be tightly packed.
    buildInputCPA.primitiveIndexOffset = ps.primitiveIndexOffset ;   // Primitive index bias, applied in optixGetPrimitiveIndex() see OptiX7Test.cu:__closesthit__ch
}

 
