
#include <cassert>
#include <iostream>

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include "sutil_vec_math.h"    // roundUp
#include "sutil_Exception.h"   // CUDA_CHECK OPTIX_CHECK


#include "Ctx.h"
#include "GAS.h"
#include "GAS_Builder.h"
#include "Engine.h"

/**
GAS_Builder::Build
--------------------

Potentially multiple bb in a single BuildInput.
Each bb could be with a CSG tree represented 
inside, so its going to need a collection of buffers.
Hmm perhaps one sbt entry per bb (or could be more).


700p17
    Acceleration structures over custom primitives are supported by referencing an
    array of primitive AABB (axis aligned bounding box) buffers in device memory,
    with one buffer per motion key. The layout of an AABB is defined in the struct
    OptixAabb. Here is an example of how to specify the build input for custom
    primitives:

    * Q:optixWhitted uses 3 bbox in one build and seems not to be doing motion ?



700p18 
    Each build input maps to one or more consecutive SBT records that control
    program dispatch.
    If multiple SBT records are required the application needs to provide a device buffer 
    with per-primitive SBT record indices for that build input. 
    If only a single SBT record is requested, 
    all primitives reference this same unique SBT record. 


Consider a set of 3 spheres of different radii, each with 
different bounds but sharing the same intersect program which 
uses the radius it reads from the SbtRecord. 




Thoughts 

* one SBT record per bb (ie per CSG solid) or one SBT for all solids within the GAS ?

* GParts is already concatenated with everything in buffers so actually 
  going with a single SBT record might be closest to OptiX6 geometry  

* how to handle variability (eg the number of nodes, transforms, planes in the trees)
  actually in the sum of the trees : all examples I havce seen have values for everything 
  there in the SBT (not pointers)

* sizes are variable for each GAS but they are known CPU side, 
  perhaps could use a CSGList template type with 4 or 5 integer template arguments 
  causing a type with appropriately sized arrays ?  For the dimensions
  of the primBuffer, tranBuffer etc...


Perhaps the best would be per-bb records and then a GAS global one
with the concatenated ?

GParts could be de-concatenated ? Hmm not so trivial probably index offsets 
to fix up.



**/

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

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

    OptixBuildInputCustomPrimitiveArray& aabbArray = buildInput.aabbArray ;  

    aabbArray.aabbBuffers   = &d_aabb_buffer;
    aabbArray.numPrimitives = num_bb ;
    aabbArray.numSbtRecords = num_bb ;   // ? 


    //unsigned flag = OPTIX_GEOMETRY_FLAG_NONE ;

    // p18 Each build input also specifies an array of OptixGeometryFlags, one for each SBT record.
    unsigned flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT ;
    unsigned* flags = new unsigned[num_bb];
    unsigned* sbt_index = new unsigned[num_bb];
    for(unsigned i=0 ; i < num_bb ; i++)
    {
        flags[i] = flag ;
        sbt_index[i] = i ; 
    } 

    aabbArray.numSbtRecords = num_bb ; 
    aabbArray.flags         = flags;
     
    // optixWhitted sets up sbtOffsets 
    // see p18 for setting up offsets 

    if(num_bb > 1)
    {
        unsigned sbt_index_size = sizeof(unsigned)*num_bb ; 
        CUdeviceptr    d_sbt_index ;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_sbt_index ), sbt_index_size ) ); 
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_sbt_index ),
                       sbt_index, sbt_index_size, 
                        cudaMemcpyHostToDevice ) ); 

        aabbArray.sbtIndexOffsetBuffer  = d_sbt_index ;
        aabbArray.sbtIndexOffsetSizeInBytes  = sizeof(unsigned);
        aabbArray.sbtIndexOffsetStrideInBytes = sizeof(unsigned);
    }

    GAS gas = Build(buildInput); 

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
    std::vector<OptixBuildInput> buildInputs ; 
    buildInputs.push_back(buildInput); 
    return Build(buildInputs) ; 
}

GAS GAS_Builder::Build(const std::vector<OptixBuildInput>& buildInputs)   // static 
{ 
    std::cout << "GAS_Builder::Build" << std::endl ;  

    GAS out = {} ; 

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = 
        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION ;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;

    OPTIX_CHECK( optixAccelComputeMemoryUsage( Ctx::context, 
                                               &accel_options, 
                                               buildInputs.data(), 
                                               buildInputs.size(), 
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

    OPTIX_CHECK( optixAccelBuild( Ctx::context,
                                  0,                  // CUDA stream
                                  &accel_options,
                                  buildInputs.data(),
                                  buildInputs.size(),                  // num build inputs
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
        OPTIX_CHECK( optixAccelCompact( Ctx::context, 
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

