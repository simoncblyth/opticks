#pragma once
 
#include "SOPTIX_BuildInput.h"

struct SOPTIX_BuildInput_IA : public SOPTIX_BuildInput
{
    static constexpr const char* NAME = "BuildInputInstanceArray" ; 

    std::vector<OptixInstance> instances ;
    CUdeviceptr instances_buffer ; 

    SOPTIX_BuildInput_IA(std::vector<OptixInstance>& _instances ); 
};


inline SOPTIX_BuildInput_IA::SOPTIX_BuildInput_IA(std::vector<OptixInstance>& _instances)
    :
    SOPTIX_BuildInput(NAME),
    instances(_instances),
    instances_buffer(0)
{
    unsigned num_bytes = sizeof( OptixInstance )*instances.size() ; 

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &instances_buffer ), num_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( instances_buffer ),
                instances.data(),
                num_bytes,
                cudaMemcpyHostToDevice
                ) );

    buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    OptixBuildInputInstanceArray& instanceArray = buildInput.instanceArray ; 
    instanceArray.instances = instances_buffer ;  
    instanceArray.numInstances = instances.size() ; 
}



