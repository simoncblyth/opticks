
#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include "sutil_vec_math.h"    // roundUp
#include "sutil_Exception.h"   // CUDA_CHECK OPTIX_CHECK


#include "GAS.h"
#include "GAS_Builder.h"
#include "Engine.h"


GAS GAS_Builder::Build(const std::vector<float>& bb )  // static
{
    unsigned num_val = bb.size() ; 
    assert( num_val % 6 == 0 );   
    unsigned num_bb = num_val / 6 ;  
    unsigned num_bytes = num_val*sizeof(float);  

    CUdeviceptr d_aabb_buffer;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_aabb_buffer ), num_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_aabb_buffer ),
                bb.data(),
                num_bytes,
                cudaMemcpyHostToDevice
                ) );

    OptixBuildInput build_input = {};

    build_input.type                    = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    build_input.aabbArray.aabbBuffers   = &d_aabb_buffer;
    build_input.aabbArray.numPrimitives = num_bb ;

    //unsigned flag = OPTIX_GEOMETRY_FLAG_NONE ;
    unsigned flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT ;
    unsigned* flags = new unsigned[num_bb];
    for(unsigned i=0 ; i < num_bb ; i++) flags[i] = flag ;

    build_input.aabbArray.flags         = flags;
    build_input.aabbArray.numSbtRecords = num_bb ;   // ? 
     
    // optixWhitted sets up sbtOffsets 


    GAS gas = Build(build_input); 

    delete[] flags ; 
    CUDA_CHECK( cudaFree( (void*)d_aabb_buffer ) );

    return gas ; 
}


/**
GAS_Builder::Build
--------------------

TODO: array of buildInput for multi-prim in single GAS
(what is the priIdx equivalent, to lookup which we intersected ?)

* presumably : unsigned int optixGetPrimitiveIndex()
* SDK/optixWhitted : 3 prim -> 3 aabb -> 1 OptixBuildInput -> 1 GAS

**/

GAS GAS_Builder::Build(OptixBuildInput buildInput)   // static 
{ 
    GAS out = {} ; 

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;

    OPTIX_CHECK( optixAccelComputeMemoryUsage( Engine::context, 
                                               &accel_options, 
                                               &buildInput, 
                                               1, 
                                               &gas_buffer_sizes 
                                             ) );
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK( cudaMalloc( 
                reinterpret_cast<void**>( &d_temp_buffer_gas ), 
                gas_buffer_sizes.tempSizeInBytes 
                ) );

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
                                  &out.handle,
                                  &emitProperty,      // emitted property list
                                  1                   // num emitted properties
                                  ) );

    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );
    //CUDA_CHECK( cudaFree( (void*)d_aabb_buffer ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &out.d_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( Engine::context, 
                                        0, 
                                        out.handle, 
                                        out.d_buffer, 
                                        compacted_gas_size, 
                                        &out.handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        out.d_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }

    return out ; 
}




