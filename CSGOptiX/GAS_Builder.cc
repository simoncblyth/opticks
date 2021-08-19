
#include <cassert>
#include <cstring>
#include <iostream>
#include <iomanip>

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include "sutil_vec_math.h"    // roundUp

#include "CUDA_CHECK.h"
#include "OPTIX_CHECK.h"

#include "Ctx.h"
#include "GAS.h"
#include "GAS_Builder.h"

#include "PLOG.hh"


/**
GAS_Builder::Build
-------------------

**/

void GAS_Builder::Build( GAS& gas, const  CSGPrimSpec& ps )  // static
{
    assert( ps.stride_in_bytes % sizeof(float) == 0 ); 
    unsigned stride_in_floats = ps.stride_in_bytes / sizeof(float) ;

    /*
    LOG(info)
        << " ps.num_prim " << std::setw(4) << ps.num_prim
        << " ps.stride_in_bytes " << ps.stride_in_bytes 
        << " ps.device " << ps.device
        << " stride_in_floats " << stride_in_floats 
        ; 

    */
    Build_11N(gas, ps);  
}

/**
GAS_Builder::Build_11N GAS:BI:AABB  1:1:N  one BI with multiple AABB
------------------------------------------------------------------------
**/

void GAS_Builder::Build_11N( GAS& gas, const CSGPrimSpec& psd )
{
    BI bi = MakeCustomPrimitivesBI_11N(psd);
    gas.bis.push_back(bi); 
    BoilerPlate(gas); 
}


/**
GAS_Builder::MakeCustomPrimitivesBI_11N
-----------------------------------------

Uploads the aabb for all prim (aka layers) of the Shape 
and arranges for separate SBT records for each prim.

**/

BI GAS_Builder::MakeCustomPrimitivesBI_11N(const CSGPrimSpec& ps)
{
    assert( ps.stride_in_bytes % sizeof(float) == 0 ); 
    unsigned stride_in_floats = ps.stride_in_bytes / sizeof(float) ;
    unsigned primitiveIndexOffset = 0 ; // offsets the normal 0,1,2,... result of optixGetPrimitiveIndex()  
    
    BI bi = {} ; 
    bi.mode = 1 ; 
    bi.flags = new unsigned[ps.num_prim];
    for(unsigned i=0 ; i < ps.num_prim ; i++) bi.flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT ; 

    assert( ps.device == true ); 

/*
    if( ps.device == false )
    {
        std::vector<float> tmp ; 
        ps.gather(tmp); 
        CSGPrimSpec::Dump(tmp); 
        LOG(error) << "YUCK : RE-UPLOADING bbox/sbtIndexOffset " ; 

        bi.sbt_index = new unsigned[ps.num_prim];
        for(unsigned i=0 ; i < ps.num_prim ; i++) bi.sbt_index[i] = i ; 

        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&bi.d_aabb), 6*sizeof(float)*ps.num_prim ));
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>(bi.d_aabb), tmp.data(), sizeof(float)*tmp.size(), cudaMemcpyHostToDevice ));

        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &bi.d_sbt_index ), sizeof(unsigned)*ps.num_prim ) ); 
        CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>(bi.d_sbt_index), bi.sbt_index, sizeof(unsigned)*ps.num_prim, cudaMemcpyHostToDevice ) ); 
    }
    else
*/
    {
        // http://www.cudahandbook.com/2013/08/why-does-cuda-cudeviceptr-use-unsigned-int-instead-of-void/ 
        // CUdeviceptr is typedef to unsigned long long 
        // uintptr_t is an unsigned integer type that is capable of storing a data pointer.

        //std::cout << "GAS_Builder::MakeCustomPrimitivesBI_11N using pre-uploaded Prim bbox/sbtIndexOffset " << std::endl ; 

        bi.d_aabb = (CUdeviceptr) (uintptr_t) ps.aabb ; 
        bi.d_sbt_index = (CUdeviceptr) (uintptr_t) ps.sbtIndexOffset ; 
    }

    bi.buildInput = {};
    bi.buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.buildInput.aabbArray ;  
    buildInputCPA.aabbBuffers = &bi.d_aabb ;  
    buildInputCPA.numPrimitives = ps.num_prim  ;   
    buildInputCPA.strideInBytes = ps.stride_in_bytes ;
    buildInputCPA.flags = bi.flags;                  // flags per sbt record
    buildInputCPA.numSbtRecords = ps.num_prim ;              // number of sbt records available to sbt index offset override. 
    buildInputCPA.sbtIndexOffsetBuffer  = bi.d_sbt_index ;   // Device pointer to per-primitive local sbt index offset buffer, Every entry must be in range [0,numSbtRecords-1]
    buildInputCPA.sbtIndexOffsetSizeInBytes  = sizeof(unsigned);  // Size of type of the sbt index offset. Needs to be 0,     1, 2 or 4    
    buildInputCPA.sbtIndexOffsetStrideInBytes = ps.stride_in_bytes ; // Stride between the index offsets. If set to zero, the offsets are assumed to be tightly packed.
    buildInputCPA.primitiveIndexOffset = primitiveIndexOffset ;  // Primitive index bias, applied in optixGetPrimitiveIndex()


/*
    std::cout 
        << " buildInputCPA.aabbBuffers[0] " 
        << " " << std::dec << buildInputCPA.aabbBuffers[0] 
        << " " << std::hex << buildInputCPA.aabbBuffers[0]  << std::dec
        << std::endl
        << " buildInputCPA.sbtIndexOffsetBuffer " 
        << " " << std::dec << buildInputCPA.sbtIndexOffsetBuffer
        << " " << std::hex << buildInputCPA.sbtIndexOffsetBuffer << std::dec
        << std::endl
        << " buildInputCPA.strideInBytes " << buildInputCPA.strideInBytes
        << " buildInputCPA.sbtIndexOffsetStrideInBytes " << buildInputCPA.sbtIndexOffsetStrideInBytes
        << std::endl
        ; 
*/  
     
    return bi ; 
} 









void GAS_Builder::DumpAABB( const float* aabb, unsigned num_aabb, unsigned stride_in_bytes )  // static 
{
    assert( stride_in_bytes % sizeof(float) == 0 ); 
    unsigned stride_in_floats = stride_in_bytes/sizeof(float); 

    std::cout 
        << "GAS_Builder::DumpAABB"
        << "  num_aabb " << num_aabb 
        << "  stride_in_bytes " << stride_in_bytes
        << "  stride_in_floats " << stride_in_floats
        << std::endl 
        ; 
    for(unsigned i=0 ; i < num_aabb ; i++)
    { 
        std::cout << std::setw(4) << i << " : " ; 
        for(unsigned j=0 ; j < 6 ; j++)  
           std::cout << std::setw(10) << std::fixed << std::setprecision(3) << *(aabb + i*stride_in_floats + j ) << " "  ; 
        std::cout << std::endl ; 
    }
}

/**
GAS_Builder::Build
---------------------

Boilerplate building the GAS from the BI vector. 
In the default 11N mode there is always only one BI in the vector.

**/

void GAS_Builder::BoilerPlate(GAS& gas)   // static 
{ 
    //std::cout << "GAS_Builder::BoilerPlate" << std::endl ;  
    unsigned num_bi = gas.bis.size() ;
    assert( num_bi == 1 ); 

    std::vector<OptixBuildInput> buildInputs ; 
    for(unsigned i=0 ; i < gas.bis.size() ; i++)
    {
        const BI& bi = gas.bis[i]; 
        buildInputs.push_back(bi.buildInput); 
        if(bi.mode == 1) assert( num_bi == 1 ); 
    }

    /*
    std::cout 
        << "GAS_Builder::BoilerPlate" 
        << " num_bi " << num_bi
        << " buildInputs.size " << buildInputs.size()
        << std::endl 
        ;  
    */

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
                                  &gas.handle,
                                  &emitProperty,      // emitted property list
                                  1                   // num emitted properties
                                  ) );

    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );
    //CUDA_CHECK( cudaFree( (void*)d_aabb_buffer ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &gas.d_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( Ctx::context, 
                                        0, 
                                        gas.handle, 
                                        gas.d_buffer, 
                                        compacted_gas_size, 
                                        &gas.handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        gas.d_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

