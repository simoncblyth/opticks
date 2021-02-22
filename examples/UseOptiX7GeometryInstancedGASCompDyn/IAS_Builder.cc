#include <iostream>
#include <iomanip>
#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include "sutil_vec_math.h"    // roundUp
#include "sutil_Exception.h"   // CUDA_CHECK OPTIX_CHECK

#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Ctx.h"
#include "IAS.h"
#include "IAS_Builder.h"
#include "Engine.h"

#include "Geo.h"

/**
**/


IAS IAS_Builder::Build( const std::vector<float>& tr ) // static 
{
    assert( tr.size() % 16 == 0 ); 
    unsigned num_tr = tr.size() / 16 ; 
    return Build( tr.data(), tr.size() ); 
}

IAS IAS_Builder::Build( const std::vector<glm::mat4>& tr ) // static 
{
    unsigned num_tr = tr.size() ; 
    std::cout << "IAS_Builder::Build num_tr " << num_tr << std::endl ; 
    return Build( (float*)tr.data(), num_tr*16 ); 
}

/**

instance.sbtOffset 
    SBT record offset.  Will only be used for instances of geometry acceleration structure (GAS) objects.
    Needs to be set to 0 for instances of instance acceleration structure (IAS) objects. 
    The maximal SBT offset can be queried using OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_SBT_OFFSET.

    See 700p43  

**/

IAS IAS_Builder::Build( const float* vals, unsigned num_vals ) // static 
{
    assert( num_vals % 16 == 0 ); 
    unsigned num_tr = num_vals / 16 ;  
 
    //unsigned flags = OPTIX_INSTANCE_FLAG_NONE ;  
    unsigned flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT ;  

    const Geo* geo = Geo::Get();  
    unsigned num_gas = geo->getNumGAS(); 
    std::cout << "IAS_Builder::Build"
              << " num_gas " << num_gas << std::endl ;   
 
    std::vector<OptixInstance> instances ;  
    for(unsigned i=0 ; i < num_tr ; i++)
    {
        void* iptr = (void*)(vals + i*16) ;  

        glm::mat4 mat(1.0f) ; 
        memcpy( glm::value_ptr(mat), iptr, 16*sizeof(float));
        
        glm::mat4 imat = glm::transpose(mat);

        glm::uvec4 idv ;
        memcpy( glm::value_ptr(idv), &imat[3], 4*sizeof(float) ); 

        unsigned instanceId = idv.x ;  
        unsigned gasIdx = idv.y ;   

        const GAS& gas = geo->getGAS(gasIdx); 

        OptixInstance instance = {} ; 
        instance.flags = flags ;

        instance.instanceId = instanceId ; 
        instance.sbtOffset = gasIdx ;           

        instance.visibilityMask = 255;
        instance.traversableHandle = gas.handle ; 

        memcpy( instance.transform, glm::value_ptr(imat), 12*sizeof( float ) );
    
        instances.push_back(instance); 
    }

    return Build(instances); 
}


IAS IAS_Builder::Build(const std::vector<OptixInstance>& instances)
{
    unsigned numInstances = instances.size() ; 
    std::cout << "IAS_Builder::bBild numInstances " << numInstances << std::endl ; 

    unsigned numBytes = sizeof( OptixInstance )*numInstances ; 

    IAS ias ; 

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

        std::cout 
            << "IAS::build (compacted is smaller) "
            << " compacted_as_size : " << compacted_as_size
            << " as_buffer_sizes.outputSizeInBytes : " << as_buffer_sizes.outputSizeInBytes
            << std::endl
            ; 

    }
    else
    {
        ias.d_buffer = d_buffer_temp_output_as_and_compacted_size;

        std::cout 
            << "IAS_Builder::Build (compacted not smaller) "
            << " compacted_as_size : " << compacted_as_size
            << " as_buffer_sizes.outputSizeInBytes : " << as_buffer_sizes.outputSizeInBytes
            << std::endl
            ; 
    }

    return ias ; 
}


