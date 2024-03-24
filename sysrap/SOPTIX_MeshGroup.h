#pragma once
/**
SOPTIX_MeshGroup.h
====================

Used from SOPTIX_Scene.h 


Here one buildInput yields one GAS.
But that is using the concatenated SMesh uploaded. 
Need to handle the sub-SMesh separately. 

Presumably using multiple buildInputs
into one GAS, because each buildInput gets its own sbtIndexOffsetBuffer.  


HMM: currently are concatenating the SMesh CPU 
side and just seeing the merged here...  but that 
is not the final solution because need to distinguish 
between landing on different sub-SMesh "CSGPrim" 

From optix7sdk.bash notes::

    CSGOptiX uses one GAS for each CSGSolid ("compound" of numPrim CSGPrim)
    and that one GAS always has only one buildInput which references
    numPrim SBT records which have "sbt-geometry-acceleration-structure-index" 
    of (0,1,2,...,numPrim-1)  


For sanity need to do something with triangles that 
follows the same pattern as that. 


Q: Can see the point of having multiple triangle BuildInputs 
   with an SBT record for each. But am mystified what use is numSbtRecords > 1 
   for a bunch of triangles ? Because no way to address them ?  

**/


#include "SCUDA_Mesh.h"
#include "SOPTIX_BuildInput_Mesh.h"
#include "SOPTIX_Accel.h"

struct SOPTIX_MeshGroup
{
    std::vector<const SOPTIX_BuildInput_Mesh*> bis ; 
    std::vector<OptixBuildInput> buildInputs ;
    SOPTIX_Accel* gas ;
 
    SOPTIX_MeshGroup( OptixDeviceContext& context, const SCUDA_Mesh* mesh ); 
    SOPTIX_MeshGroup( OptixDeviceContext& context, std::vector<const SCUDA_Mesh*>& meshes ); 
    void init(  OptixDeviceContext& context, const SCUDA_Mesh** meshes, int num_mesh ); 

    std::string desc() const ; 
}; 

inline std::string SOPTIX_MeshGroup::desc() const 
{
    int num_bi = bis.size() ; 
    std::stringstream ss ; 
    ss << "[SOPTIX_MeshGroup::desc num_bi "  << num_bi << std::endl ; 
    for(int i=0 ; i < num_bi ; i++)
    {
        const SOPTIX_BuildInput_Mesh* bi = bis[i] ; 
        ss << "bi[" << i << "]" << std::endl ; 
        ss << bi->desc() << std::endl ; 
    }
    ss << gas->desc() << std::endl ;   
    ss << "]SOPTIX_MeshGroup::desc num_bi "  << num_bi << std::endl ; 
    std::string str = ss.str() ; 
    return str ; 
}


inline SOPTIX_MeshGroup::SOPTIX_MeshGroup(OptixDeviceContext& context, const SCUDA_Mesh* meshes )
{
    init( context, meshes, 1 ); 
}
inline SOPTIX_MeshGroup::SOPTIX_MeshGroup(OptixDeviceContext& context, std::vector<const SCUDA_Mesh*>& meshes )
{
    init( context, meshes.data(), meshes.size() ); 
}

inline SOPTIX_MeshGroup::init( OptixDeviceContext& context, const SCUDA_Mesh** meshes, int num_mesh )
{
    for(int i=0 ; i < num_mesh ; i++)
    {
        const SCUDA_Mesh* mesh = meshes[i]; 
        const SOPTIX_BuildInput_Mesh* bi = new SOPTIX_BuildInput_Mesh(mesh); 
        bis.push_back(bi); 
        buildInputs.push_back(bi->buildInput); 
    }
    gas = new SOPTIX_Accel( context, buildInputs );     
}

