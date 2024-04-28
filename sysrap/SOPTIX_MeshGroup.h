#pragma once
/**
SOPTIX_MeshGroup.h : create SOPTIX_BuildInput_Mesh for each part of SCUDA_MeshGroup, use those to form SOPTIX_Accel gas  
========================================================================================================================

Used from SOPTIX_Scene.h 

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
    OptixDeviceContext& context ; 
    const SCUDA_MeshGroup* cmg ; 

    std::vector<const SOPTIX_BuildInput_Mesh*> bis ; 
    std::vector<OptixBuildInput> buildInputs ;
    SOPTIX_Accel* gas ;

    std::string desc() const ; 
    size_t num_buildInputs() const ; 

    SOPTIX_MeshGroup( OptixDeviceContext& context, const SCUDA_MeshGroup* cmg ); 
    void init(); 
    static SOPTIX_MeshGroup* Create( OptixDeviceContext& ctx, const SMeshGroup* mg );  // more vertical API
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

inline size_t SOPTIX_MeshGroup::num_buildInputs() const
{
    assert( buildInputs.size() == bis.size() ) ; 
    return buildInputs.size() ; 
}

/**
SOPTIX_MeshGroup::SOPTIX_MeshGroup
-----------------------------------


**/

inline SOPTIX_MeshGroup::SOPTIX_MeshGroup(OptixDeviceContext& _context, const SCUDA_MeshGroup* _cmg )
    :
    context(_context),
    cmg(_cmg)
{
    init();
}

inline void SOPTIX_MeshGroup::init()
{
    size_t num_part = cmg->num_part() ;  
    for(size_t i=0 ; i < num_part ; i++)
    {
        const SOPTIX_BuildInput_Mesh* bi = new SOPTIX_BuildInput_Mesh(cmg, i); 
        bis.push_back(bi); 
        buildInputs.push_back(bi->buildInput); 
    }
    gas = new SOPTIX_Accel( context, buildInputs );     
}


/**
SOPTIX_MeshGroup::Create
-------------------------

This one-step API from CPU side SMeshGroup (from SScene) to GPU side SOPTIX_MeshGroup
is to facilitate tri/ana integration. 

**/

inline SOPTIX_MeshGroup* SOPTIX_MeshGroup::Create( OptixDeviceContext& ctx, const SMeshGroup* mg )
{
    SCUDA_MeshGroup* cmg = SCUDA_MeshGroup::Upload(mg) ; 
    SOPTIX_MeshGroup* xmg = new SOPTIX_MeshGroup( ctx, cmg ); 
    return xmg ;  
}


