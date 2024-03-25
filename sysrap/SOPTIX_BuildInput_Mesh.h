#pragma once
/**
SOPTIX_BuildInput_Mesh.h
==========================

Intances of this need to be kept alive whilst 
using the OptixBuildInput because of the 
vertexBuffers field that is a host pointer 
to device arrays. Have to keep that address valid.  

**/
#include "SCUDA_MeshGroup.h"

struct SOPTIX_BuildInput_Mesh
{
    unsigned    flag ;  
    CUdeviceptr vertexBuffer ;  
    CUdeviceptr indexBuffer ; 
    size_t vtx_elem ; 
    size_t idx_elem ; 
 
    OptixBuildInput buildInput = {} ;
 
    SOPTIX_BuildInput_Mesh( const SCUDA_MeshGroup* _mg, size_t part ); 
    std::string desc() const ; 
}; 

inline SOPTIX_BuildInput_Mesh::SOPTIX_BuildInput_Mesh( const SCUDA_MeshGroup* mg, size_t part )
    :
    flag( OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT | OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING ),
    vertexBuffer( mg->vtx.pointer(part) ),
    indexBuffer( mg->idx.pointer(part) ),
    vtx_elem( mg->vtx.item_num(part) ),
    idx_elem( mg->idx.item_num(part) )
{
    assert( vtx_elem % 3 == 0 ); 
    assert( idx_elem % 3 == 0 ); 
    unsigned numVertices = vtx_elem/3 ; 
    unsigned numIndexTriplets = idx_elem/3 ; 

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


    triangleArray.vertexBuffers = vertexBuffers ; // only one without motion
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


inline std::string SOPTIX_BuildInput_Mesh::desc() const 
{
    std::stringstream ss ; 
    ss << "[SOPTIX_BuildInput_Mesh::desc" << std::endl ; 
    ss <<  SOPTIX::DescBuildInputTriangleArray(buildInput) ; 
    ss << "]SOPTIX_BuildInput_Mesh::desc" << std::endl ; 
    std::string str = ss.str() ; 
    return str ; 
}

