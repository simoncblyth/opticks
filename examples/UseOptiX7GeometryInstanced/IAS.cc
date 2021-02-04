
#include <iostream>
#include <iomanip>
#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include "sutil_vec_math.h"    // roundUp
#include "sutil_Exception.h"   // CUDA_CHECK OPTIX_CHECK


#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "IAS.h"
#include "Engine.h"


IAS::IAS(OptixTraversableHandle gas_handle_) 
    :
    gas_handle(gas_handle_)
{
    init(); 
}


/**
Following example of RenderCore::UpdateToplevel
   /home/blyth/local/env/graphics/lighthouse2/lighthouse2/lib/rendercore_optix7/rendercore.cpp 

**/


void IAS::dump(const float* imp)
{
    for(int i=0 ; i < 3 ; i++ )
    {
        for(int j=0 ; j < 4 ; j++ ) std::cout << " " << std::setw(10) << imp[i*4+j] ;  
        std::cout << std::endl ; 
    }
}

/**
IAS::addInstance
-----------------

Collect instance transforms 

TODO: accept an array of transforms 
and do it all at once, keeping the optix types 
out of the interface

**/

void IAS::addInstance(const glm::mat4& mat)
{
    unsigned idx = instances.size() ; 

    glm::mat4 imat = glm::transpose(mat);
    const float* imp = glm::value_ptr(imat); 


    OptixInstance instance = {} ; 

    //instance.flags = OPTIX_INSTANCE_FLAG_NONE;
    instance.flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT ;
    instance.instanceId = idx ; 
    instance.sbtOffset = 0 ; 
    instance.visibilityMask = 255;
    instance.traversableHandle = gas_handle ; 

    memcpy( instance.transform, imp, 12 * sizeof( float ) );
    //dump(instance.transform);  

    instances.push_back(instance); 
}


void IAS::init()
{
    //initInstancesOne(); 
    //initInstancesTwo(); 
    initInstancesMany(); 

    build(); 
}

void IAS::initInstancesOne()
{
    glm::mat4 identity(1.f) ; 
    addInstance(identity); 
}

void IAS::initInstancesTwo()
{
    glm::vec3 a_tlat(0.f,0.f,0.5f) ; 
    glm::mat4 a_tr(1.f) ;
    a_tr = glm::translate(a_tr, a_tlat );
 
    glm::vec3 b_tlat(0.f,0.5f,0.f) ; 
    glm::mat4 b_tr(1.f) ;
    b_tr = glm::translate(b_tr, b_tlat );
 
    addInstance(a_tr); 
    addInstance(b_tr); 
}

void IAS::initInstancesMany()
{
    //int n=100 ;   // 8,120,601
    int n=50 ;   // 
    int s=1 ; 

    for(int i=-n ; i <= n ; i+=s ){
    for(int j=-n ; j <= n ; j+=s ){
    for(int k=-n ; k <= n ; k+=s ){

        glm::vec3 tlat(i*1.f,j*1.f,k*1.f) ; 
        glm::mat4 tr(1.f) ;
        tr = glm::translate(tr, tlat );
        addInstance(tr); 

    }
    }
    }
}


void IAS::build()
{
    unsigned numInstances = instances.size() ; 
    std::cout << "IAS::build numInstances " << numInstances << std::endl ; 

    unsigned numBytes = sizeof( OptixInstance )*numInstances ; 

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_instances ), numBytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_instances ),
                instances.data(),
                numBytes,
                cudaMemcpyHostToDevice
                ) );

 
    OptixBuildInput buildInput = {};

    buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    buildInput.instanceArray.instances = d_instances ; 
    buildInput.instanceArray.numInstances = numInstances ; 

    ias_handle = build(buildInput); 
}


OptixTraversableHandle IAS::build(OptixBuildInput buildInput)
{
    OptixTraversableHandle handle;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes as_buffer_sizes;

    OPTIX_CHECK( optixAccelComputeMemoryUsage( Engine::context, &accel_options, &buildInput, 1, &as_buffer_sizes ) );
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


    OPTIX_CHECK( optixAccelBuild( Engine::context,
                                  0,                  // CUDA stream
                                  &accel_options,
                                  &buildInput,
                                  1,                  // num build inputs
                                  d_temp_buffer_as,
                                  as_buffer_sizes.tempSizeInBytes,
                                  d_buffer_temp_output_as_and_compacted_size,
                                  as_buffer_sizes.outputSizeInBytes,
                                  &handle,
                                  &emitProperty,      // emitted property list
                                  1                   // num emitted properties
                                  ) );

    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_as ) );

    size_t compacted_as_size;
    CUDA_CHECK( cudaMemcpy( &compacted_as_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );


    if( compacted_as_size < as_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_as_output_buffer ), compacted_as_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( Engine::context, 0, handle, d_as_output_buffer, compacted_as_size, &handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_as_and_compacted_size ) );

        std::cout 
            << "IAS::build (compacted is smaller) "
            << " compacted_as_size : " << compacted_as_size
            << " as_buffer_sizes.outputSizeInBytes : " << as_buffer_sizes.outputSizeInBytes
            << std::endl
            ; 

    }
    else
    {
        d_as_output_buffer = d_buffer_temp_output_as_and_compacted_size;

        std::cout 
            << "IAS::build (compacted not smaller) "
            << " compacted_as_size : " << compacted_as_size
            << " as_buffer_sizes.outputSizeInBytes : " << as_buffer_sizes.outputSizeInBytes
            << std::endl
            ; 
    }

    return handle ; 
}


