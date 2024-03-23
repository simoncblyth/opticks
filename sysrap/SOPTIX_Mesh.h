#pragma once
/**
SOPTIX_Mesh.h
===============

Used from SOPTIX_Scene.h 

**/


#include "SCUDA_Mesh.h"
#include "SOPTIX_Accel.h"

struct SOPTIX_Mesh
{
    SOPTIX* ox ; 
    const SCUDA_Mesh* mesh ;

    unsigned    flag ;  
    CUdeviceptr vertexBuffer ;  
    CUdeviceptr indexBuffer  ; 
    OptixBuildInput buildInput = {} ;
    SOPTIX_Accel* gas ;
 
    SOPTIX_Mesh(SOPTIX* _ox,  const SCUDA_Mesh* _mesh ); 

    void init(); 
    void initBuildInput(); 
    void initGAS(); 

    std::string desc() const ; 
    std::string descBuildInput() const ; 
    std::string descGAS() const ; 
}; 

inline std::string SOPTIX_Mesh::desc() const 
{
    std::stringstream ss ; 
    ss << descBuildInput() ;
    ss << descGAS() ;
    std::string str = ss.str() ; 
    return str ; 
}

inline std::string SOPTIX_Mesh::descBuildInput() const 
{
    std::stringstream ss ; 
    ss << "[SOPTIX_Mesh::descBuildInput" << std::endl ; 
    ss <<  SOPTIX::DescBuildInputTriangleArray(buildInput) ; 
    ss << "]SOPTIX_Mesh::descBuildInput" << std::endl ; 
    std::string str = ss.str() ; 
    return str ; 
}

inline std::string SOPTIX_Mesh::descGAS() const 
{
    std::stringstream ss ; 
    ss << "[SOPTIX_Mesh::descGAS" << std::endl ; 
    ss << gas->desc() << std::endl ;   
    ss << "]SOPTIX_Mesh::descGAS" << std::endl ; 
    std::string str = ss.str() ; 
    return str ; 
}


inline SOPTIX_Mesh::SOPTIX_Mesh(SOPTIX* _ox, const SCUDA_Mesh* _mesh )
    :
    ox(_ox),
    mesh(_mesh),
    flag( OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT | OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING ),
    vertexBuffer(  mesh->vtx.pointer() ),
    indexBuffer(   mesh->idx.pointer() ),
    gas(nullptr)
{
    init();
}


inline void SOPTIX_Mesh::init()
{
    initBuildInput(); 
    initGAS();
}

/**
SOPTIX_Mesh::initBuildInput
-----------------------------

HMM: currently are concatenating the SMesh CPU 
side and presenting the merged here...  but that 
is not the final solution because need to distinguish 
between landing on different sub-SMesh "CSGPrim" 

From optix7sdk.bash notes::

    CSGOptiX uses one GAS for each CSGSolid ("compound" of numPrim CSGPrim)
    and that one GAS always has only one buildInput which references
    numPrim SBT records which have "sbt-geometry-acceleration-structure-index" 
    of (0,1,2,...,numPrim-1)  


For sanity need to do something with triangles that 
follows the same pattern as that. 

**/

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

    // TODO:
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
    gas = new SOPTIX_Accel( ox->context, buildInputs );     
}


