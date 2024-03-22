#pragma once
/**
SOPTIX_SBT.h
==================

**/

#include "SOPTIX_Binding.h"

struct SOPTIX_SBT
{ 
    SOPTIX_Pipeline& pip ;     

    CUdeviceptr   d_raygen ;
    CUdeviceptr   d_miss ;
    CUdeviceptr   d_hitgroup ;

    OptixShaderBindingTable sbt = {};  

    SOPTIX_SBT( SOPTIX_Pipeline& pip ); 

    void init(); 
    void initRaygen(); 
    void initMiss(); 
    void initHitgroup(); 
};


inline SOPTIX_SBT::SOPTIX_SBT(
       SOPTIX_Pipeline& _pip
    )
    :
    pip(_pip)
{
    init(); 
}
 
inline void SOPTIX_SBT::init()
{
    initRaygen(); 
    initMiss(); 
    initHitgroup(); 
}

inline void SOPTIX_SBT::initRaygen()
{
    const size_t raygen_record_size = sizeof( SOPTIX_EmptyRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &sbt.raygenRecord ), raygen_record_size ) );
    
    SOPTIX_EmptyRecord rg_sbt;
    OPTIX_CHECK( optixSbtRecordPackHeader( pip.raygen_pg, &rg_sbt ) );

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( sbt.raygenRecord ),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice
                ) );

}

inline void SOPTIX_SBT::initMiss()
{
    const size_t miss_record_size = sizeof( SOPTIX_EmptyRecord );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &sbt.missRecordBase ), miss_record_size ) );
    
    SOPTIX_EmptyRecord ms_sbt;
    OPTIX_CHECK( optixSbtRecordPackHeader( pip.miss_pg, &ms_sbt ) );

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( sbt.missRecordBase ),
                &ms_sbt,
                miss_record_size,
                cudaMemcpyHostToDevice
                ) );

    sbt.missRecordStrideInBytes = static_cast<uint32_t>( miss_record_size );
    sbt.missRecordCount = 1 ; 
}


inline void SOPTIX_SBT::initHitgroup()
{
    // HMM : NEED TO DECIDE WHATS NEEDED GPU SIDE BEFORE CAN DO THIS
}




