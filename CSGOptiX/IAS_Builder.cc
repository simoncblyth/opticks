#include <cassert>
#include <iostream>
#include <iomanip>
#include <map>
#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include "scuda.h"    // roundUp

#include "CUDA_CHECK.h"
#include "OPTIX_CHECK.h"

#include "Ctx.h"
#include "Properties.h"

#include "GAS.h"
#include "IAS.h"
#include "IAS_Builder.h"
#include "SBT.h"


/**
IAS_Builder::Build
-------------------

Boilerplate turning the vector of OptixInstance into an IAS.

**/

void IAS_Builder::Build(IAS& ias, const std::vector<OptixInstance>& instances)
{
    unsigned numInstances = instances.size() ; 

    unsigned numBytes = sizeof( OptixInstance )*numInstances ; 


    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &ias.d_instances ), numBytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( ias.d_instances ),
                instances.data(),
                numBytes,
                cudaMemcpyHostToDevice
                ) );

 
    OptixBuildInput buildInput = {};

    buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    buildInput.instanceArray.instances = ias.d_instances ; 
    buildInput.instanceArray.numInstances = numInstances ; 


    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = 
        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION ;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes as_buffer_sizes;

    OPTIX_CHECK( optixAccelComputeMemoryUsage( Ctx::context, &accel_options, &buildInput, 1, &as_buffer_sizes ) );
    CUdeviceptr d_temp_buffer_as;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer_as ), as_buffer_sizes.tempSizeInBytes ) );

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_as_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>( as_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_buffer_temp_output_as_and_compacted_size ),
                compactedSizeOffset + 8
                ) );

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result             = ( CUdeviceptr )( (char*)d_buffer_temp_output_as_and_compacted_size + compactedSizeOffset );


    OPTIX_CHECK( optixAccelBuild( Ctx::context,
                                  0,                  // CUDA stream
                                  &accel_options,
                                  &buildInput,
                                  1,                  // num build inputs
                                  d_temp_buffer_as,
                                  as_buffer_sizes.tempSizeInBytes,
                                  d_buffer_temp_output_as_and_compacted_size,
                                  as_buffer_sizes.outputSizeInBytes,
                                  &ias.handle,
                                  &emitProperty,      // emitted property list
                                  1                   // num emitted properties
                                  ) );

    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_as ) );

    size_t compacted_as_size;
    CUDA_CHECK( cudaMemcpy( &compacted_as_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );


    if( compacted_as_size < as_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &ias.d_buffer ), compacted_as_size ) );

        // use ias.handle as input and output
        OPTIX_CHECK( optixAccelCompact( Ctx::context, 0, ias.handle, ias.d_buffer, compacted_as_size, &ias.handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_as_and_compacted_size ) );

        std::cerr 
            << "(compacted is smaller) "
            << " compacted_as_size : " << compacted_as_size
            << " as_buffer_sizes.outputSizeInBytes : " << as_buffer_sizes.outputSizeInBytes
            << "\n"  
            ; 

    }
    else
    {
        ias.d_buffer = d_buffer_temp_output_as_and_compacted_size;

        std::cerr 
            << "(compacted not smaller) "
            << " compacted_as_size : " << compacted_as_size
            << " as_buffer_sizes.outputSizeInBytes : " << as_buffer_sizes.outputSizeInBytes
            << "\n"  
            ; 
    }
}


