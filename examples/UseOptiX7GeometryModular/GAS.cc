
#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include "sutil_vec_math.h"    // roundUp
#include "sutil_Exception.h"   // CUDA_CHECK OPTIX_CHECK
#include "GAS.h"
#include "Engine.h"


GAS::GAS()
{
    init(); 
}

void GAS::init()
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    // AABB build input
    OptixAabb   aabb = {-1.5f, -1.5f, -1.5f, 1.5f, 1.5f, 1.5f};
    CUdeviceptr d_aabb_buffer;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb_buffer ), sizeof( OptixAabb ) ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_aabb_buffer ),
                &aabb,
                sizeof( OptixAabb ),
                cudaMemcpyHostToDevice
                ) );

    OptixBuildInput buildInput = {};

    buildInput.type                    = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    buildInput.customPrimitiveArray.aabbBuffers   = &d_aabb_buffer;
    buildInput.customPrimitiveArray.numPrimitives = 1;
    // only one as no motion 

    uint32_t aabb_input_flags[1]       = {OPTIX_GEOMETRY_FLAG_NONE};
    buildInput.customPrimitiveArray.flags         = aabb_input_flags;
    buildInput.customPrimitiveArray.numSbtRecords = 1;


    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( Engine::context, &accel_options, &buildInput, 1, &gas_buffer_sizes ) );
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer_gas ), gas_buffer_sizes.tempSizeInBytes ) );

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
                compactedSizeOffset + 8
                ) );

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result             = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

    OPTIX_CHECK( optixAccelBuild( Engine::context,
                                  0,                  // CUDA stream
                                  &accel_options,
                                  &buildInput,
                                  1,                  // num build inputs
                                  d_temp_buffer_gas,
                                  gas_buffer_sizes.tempSizeInBytes,
                                  d_buffer_temp_output_gas_and_compacted_size,
                                  gas_buffer_sizes.outputSizeInBytes,
                                  &gas_handle,
                                  &emitProperty,      // emitted property list
                                  1                   // num emitted properties
                                  ) );

    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );
    CUDA_CHECK( cudaFree( (void*)d_aabb_buffer ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_gas_output_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( Engine::context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}



