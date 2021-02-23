
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

    std::cout << "GAS_Builder::Build num_val " << num_val << " num_bb " << num_bb << std::endl ;  

    std::vector<BI> bis ; 
    for(unsigned i=0 ; i < num_bb ; i++)
    { 
         const float* ptr = bb.data() + i*6u ; 
         BI bi = MakeCustomPrimitivesBI( ptr, 6u );  
         bis.push_back(bi); 
    }

    std::cout << "GAS_Builder::Build bis.size " << bis.size() << std::endl ; 

    GAS gas = Build(bis); 
    return gas ; 
}


BI GAS_Builder::MakeCustomPrimitivesBI(const float* bb, unsigned num_val )
{
    assert( num_val == 6 ); 

    std::cout << "GAS_Builder::MakeCustomPrimitivesBI " ; 
    for(unsigned i=0 ; i < 6 ; i++) std::cout << *(bb+i) << " "  ; 
    std::cout << std::endl ;  

    BI bi = {} ; 
    bi.buildInput = {};
    bi.buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    bi.num_sbt_records = 1 ;    //  SBT entries for each build input
    bi.flags = new unsigned[bi.num_sbt_records];
    bi.sbt_index = new unsigned[bi.num_sbt_records];
    bi.flags[0] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT ; // p18: Each build input also specifies an array of OptixGeometryFlags, one for each SBT record.
    bi.sbt_index[0] = 0 ; 

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &bi.d_aabb ), 6*sizeof(float) ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( bi.d_aabb ),
                            bb, 6*sizeof(float),
                            cudaMemcpyHostToDevice ));

    OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.buildInput.aabbArray ;  
    buildInputCPA.aabbBuffers = &bi.d_aabb ;  
    buildInputCPA.numPrimitives = 1 ;   
    buildInputCPA.numSbtRecords = bi.num_sbt_records ;  
    buildInputCPA.flags = bi.flags;
     
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &bi.d_sbt_index ), sizeof(unsigned)*bi.num_sbt_records ) ); 
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( bi.d_sbt_index ),
                            bi.sbt_index, sizeof(unsigned)*bi.num_sbt_records, 
                            cudaMemcpyHostToDevice ) ); 

    buildInputCPA.sbtIndexOffsetBuffer  = bi.d_sbt_index ;
    buildInputCPA.sbtIndexOffsetSizeInBytes  = sizeof(unsigned);
    buildInputCPA.sbtIndexOffsetStrideInBytes = sizeof(unsigned);
    buildInputCPA.primitiveIndexOffset = 0 ;  // Primitive index bias, applied in optixGetPrimitiveIndex()

    return bi ; 
} 



GAS GAS_Builder::Build(const std::vector<BI>& bis)   // static 
{ 
    std::cout << "GAS_Builder::Build" << std::endl ;  

    GAS out = {} ; 
    out.num_sbt_rec = bis.size() ;  
    out.bis = bis ; 

    std::vector<OptixBuildInput> buildInputs ; 
    for(unsigned i=0 ; i < bis.size() ; i++)
    {
        const BI& bi = bis[i]; 
        buildInputs.push_back(bi.buildInput); 
    }

    std::cout 
        << "GAS_Builder::Build" 
        << " bis.size " << bis.size()
        << " buildInputs.size " << buildInputs.size()
        << std::endl 
        ;  
    

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
