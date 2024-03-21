#pragma once

#include "SCUDA_Mesh.h"

struct SOPTIX_Mesh
{
    SOPTIX* ox ; 
    const SCUDA_Mesh* mesh ;

    unsigned    flag ;  
    CUdeviceptr vertexBuffer ;  
    CUdeviceptr indexBuffer  ; 
    OptixBuildInput buildInput = {} ;

    CUdeviceptr gas_buffer ; 
    OptixTraversableHandle gas_handle ; 

    
    SOPTIX_Mesh(SOPTIX* _ox,  const SCUDA_Mesh* _mesh ); 

    void init(); 
    void initBuildInput(); 
    void initGAS(); 

    std::string desc() const ; 
}; 

inline std::string SOPTIX_Mesh::desc() const 
{
    std::stringstream ss ; 
    ss << "[SOPTIX_Mesh::desc" << std::endl ; 
    ss <<  SOPTIX::DescBuildInputTriangleArray(buildInput) ; 
    ss << "]SOPTIX_Mesh::desc" << std::endl ; 
    std::string str = ss.str() ; 
    return str ; 
}

inline SOPTIX_Mesh::SOPTIX_Mesh(SOPTIX* _ox, const SCUDA_Mesh* _mesh )
    :
    ox(_ox),
    mesh(_mesh),
    flag( OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT | OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING ),
    vertexBuffer(  mesh->vtx.pointer() ),
    indexBuffer(   mesh->idx.pointer() ) 
{
    init();
}


inline void SOPTIX_Mesh::init()
{
    initBuildInput(); 
    initGAS();
}

inline void SOPTIX_Mesh::initBuildInput()
{ 
    assert( mesh->vtx.num_item % 3 == 0 ); 
    assert( mesh->idx.num_item % 3 == 0 ); 

    unsigned numVertices = mesh->vtx.num_item/3 ; 
    unsigned numIndexTriplets = mesh->idx.num_item/3 ; 

    unsigned vertexStrideInBytes = 0 ; 

    OptixBuildInputType type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES ;
    OptixVertexFormat vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3 ; 
    OptixIndicesFormat indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3 ;  

    unsigned indexStrideInBytes = 0 ; 
    CUdeviceptr preTransform = 0 ; 
    OptixTransformFormat transformFormat = OPTIX_TRANSFORM_FORMAT_NONE ;  

    buildInput.type = type ; 

    OptixBuildInputTriangleArray& triangleArray = buildInput.triangleArray ;

    unsigned numSbtRecords = 1 ; 
    CUdeviceptr sbtIndexOffsetBuffer = 0 ; 
    unsigned sbtIndexOffsetSizeInBytes = 0 ; 
    unsigned sbtIndexOffsetStrideInBytes = 0 ; 
    unsigned primitiveIndexOffset = 0 ; 

    // NB made these members to keep the addresses valid
    const unsigned* flags = &flag ; 
    CUdeviceptr* vertexBuffers = &vertexBuffer  ; 


    triangleArray.vertexBuffers = vertexBuffers ;
    triangleArray.numVertices = numVertices ;  
    triangleArray.vertexFormat = vertexFormat ; 
    triangleArray.vertexStrideInBytes = vertexStrideInBytes ; 
    triangleArray.indexBuffer = indexBuffer ; 
    triangleArray.numIndexTriplets = numIndexTriplets ; 
    triangleArray.indexFormat = indexFormat ;     
    triangleArray.indexStrideInBytes = indexStrideInBytes ; 
    triangleArray.preTransform = preTransform ; 
    triangleArray.flags = flags ; 
    triangleArray.numSbtRecords = numSbtRecords ;    
    triangleArray.sbtIndexOffsetBuffer = sbtIndexOffsetBuffer ; 
    triangleArray.sbtIndexOffsetSizeInBytes = sbtIndexOffsetSizeInBytes ; 
    triangleArray.sbtIndexOffsetStrideInBytes = sbtIndexOffsetStrideInBytes ;
    triangleArray.primitiveIndexOffset = primitiveIndexOffset ; 
    triangleArray.transformFormat = transformFormat ;     
} 


inline void SOPTIX_Mesh::initGAS()
{
    std::vector<OptixBuildInput> buildInputs ;
    buildInputs.push_back( buildInput ); 

    OptixAccelBuildOptions accel_options = {};

    unsigned _buildFlags = 0 ; 
    _buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE  ; 
    _buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION ; 
    //_buildFlags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS ; // see optixGetTriangleVertexData
    //_buildFlags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS ; // see optixGetInstanceTraversableFromIAS

    OptixBuildFlags buildFlags = (OptixBuildFlags)_buildFlags ;
    OptixBuildOperation operation = OPTIX_BUILD_OPERATION_BUILD ; 
    OptixMotionOptions motionOptions = {} ; 

    accel_options.buildFlags = buildFlags ; 
    accel_options.operation  = operation ;
    accel_options.motionOptions = motionOptions ; 

    OptixAccelBufferSizes accelBufferSizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( ox->context, 
                                               &accel_options, 
                                               buildInputs.data(),
                                               buildInputs.size(),
                                               &accelBufferSizes
                                             ) );
     
    std::cout << SOPTIX::DescAccelBufferSizes(accelBufferSizes) ;  


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

    OPTIX_CHECK( optixAccelBuild( ox->context,
                                  stream,   
                                  accelOptions,
                                  buildInputs.data(),
                                  buildInputs.size(),                  // num build inputs
                                  tempBuffer,
                                  accelBufferSizes.tempSizeInBytes,
                                  outputBuffer,
                                  accelBufferSizes.outputSizeInBytes,
                                  &gas_handle,
                                  &emitProperty,      
                                  numEmittedProperties  
                                  ) );


    CUDA_CHECK( cudaFree( (void*)tempBuffer ) ); 

    size_t compacted_size;
    CUDA_CHECK( cudaMemcpy( &compacted_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );



    if( compacted_size < accelBufferSizes.outputSizeInBytes )
    {
        std::cout 
            << " compacted_size " << compacted_size 
            << " PROCEED WITH COMPACTING " 
            << std::endl 
            ; 
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &gas_buffer ), compacted_size ) );

        // use gas_handle as input and output
        OPTIX_CHECK( optixAccelCompact( ox->context,
                                        stream,
                                        gas_handle,
                                        gas_buffer,
                                        compacted_size,
                                        &gas_handle ) );

        CUDA_CHECK( cudaFree( (void*)outputBuffer ) );
    }
    else
    {
        std::cout 
            << " compacted_size " << compacted_size 
            << " SKIP COMPACTING " 
            << std::endl 
            ; 
        gas_buffer = outputBuffer ; 
    }
}




