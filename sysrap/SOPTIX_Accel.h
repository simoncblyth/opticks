#pragma once
/**
SOPTIX_Accel.h : builds acceleration structure GAS or IAS from the buildInputs
===============================================================================

Used by: SOPTIX_Mesh.h 

**/

struct SOPTIX_Accel
{
    CUdeviceptr buffer ; 
    OptixTraversableHandle handle ; 

    OptixBuildFlags buildFlags = {} ; 
    OptixAccelBufferSizes accelBufferSizes = {} ;
    size_t compacted_size ;
    bool   compacted ; 

    SOPTIX_Accel( OptixDeviceContext& context, const std::vector<OptixBuildInput>& buildInputs );  
    std::string desc() const ;
};


inline std::string SOPTIX_Accel::desc() const
{
    std::stringstream ss ; 
    ss << "[SOPTIX_Accel::desc " << std::endl ; 
    ss << " compacted " << ( compacted ? "YES" : "NO " ) << std::endl ;
    ss << " compacted_size " << compacted_size << std::endl ;
    ss << SOPTIX_Desc::AccelBufferSizes(accelBufferSizes) ;  
    ss << "]SOPTIX_Accel::desc " << std::endl ; 
    std::string str = ss.str(); 
    return str ; 
}


inline SOPTIX_Accel::SOPTIX_Accel( OptixDeviceContext& context, const std::vector<OptixBuildInput>& buildInputs )     
    :
    buffer(0), 
    handle(0), 
    compacted_size(0),
    compacted(false)
{ 
    OptixAccelBuildOptions accel_options = {};

    unsigned _buildFlags = 0 ; 
    _buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE  ; 
    _buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION ; 
    //_buildFlags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS ; // see optixGetTriangleVertexData
    //_buildFlags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS ; // see optixGetInstanceTraversableFromIAS

    buildFlags = (OptixBuildFlags)_buildFlags ;

    OptixBuildOperation operation = OPTIX_BUILD_OPERATION_BUILD ; 
    OptixMotionOptions motionOptions = {} ; 

    accel_options.buildFlags = buildFlags ; 
    accel_options.operation  = operation ;
    accel_options.motionOptions = motionOptions ; 

    OPTIX_CHECK( optixAccelComputeMemoryUsage( context, 
                                               &accel_options, 
                                               buildInputs.data(),
                                               buildInputs.size(),
                                               &accelBufferSizes
                                             ) );
     


    CUstream stream = 0 ;   
    const OptixAccelBuildOptions* accelOptions = &accel_options ; 

    CUdeviceptr tempBuffer ;
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &tempBuffer ),
                accelBufferSizes.tempSizeInBytes
                ) );

    size_t      compactedSizeOffset = roundUp<size_t>( accelBufferSizes.outputSizeInBytes, 8ull );

    // expand the outputBuffer size by 8 bytes plus any 8-byte alignment padding
    // to give somewhere to write the emitted property

    CUdeviceptr outputBuffer ;
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &outputBuffer),
                compactedSizeOffset + 8 
                ) );


    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type    = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result  = ( CUdeviceptr )( (char*)outputBuffer + compactedSizeOffset );
    unsigned numEmittedProperties = 1 ; 

    OPTIX_CHECK( optixAccelBuild( context,
                                  stream,   
                                  accelOptions,
                                  buildInputs.data(),
                                  buildInputs.size(),
                                  tempBuffer,
                                  accelBufferSizes.tempSizeInBytes,
                                  outputBuffer,
                                  accelBufferSizes.outputSizeInBytes,
                                  &handle,
                                  &emitProperty,      
                                  numEmittedProperties  
                                  ) );


    CUDA_CHECK( cudaFree( (void*)tempBuffer ) ); 

    CUDA_CHECK( cudaMemcpy( &compacted_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

    if( compacted_size < accelBufferSizes.outputSizeInBytes )
    {
        compacted = true ; 
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &buffer ), compacted_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( context,
                                        stream,
                                        handle,
                                        buffer,
                                        compacted_size,
                                        &handle ) );

        CUDA_CHECK( cudaFree( (void*)outputBuffer ) );
    }
    else
    {
        compacted = false ; 
        buffer = outputBuffer ; 
    }
}

