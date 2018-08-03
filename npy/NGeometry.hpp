#pragma once

#include "NPY_API_EXPORT.hh"
#include <vector>
#include <string>
#include <glm/fwd.hpp>
struct nd ; 
class NParameters ; 
class NCSG ; 
struct NSceneConfig ; 

/**
NGeometry
===========

Pure virtual interface fulfilled by:

1. NGLTF 


**/


class NPY_API NGeometry
{
    public:
       virtual std::string              desc() const = 0 ;  
       virtual unsigned                 getNumNodes()  const = 0 ; 
       virtual unsigned                 getNumMeshes() const = 0 ; 
       virtual const std::vector<int>&  getNodeChildren(unsigned idx) const  = 0 ; 
       virtual nd*                      createNdConverted(unsigned idx, unsigned depth, nd* parent) const  = 0 ; 
       virtual void                     compare_trees_r(unsigned idx) = 0 ; 
       virtual unsigned                 getSourceVerbosity() = 0 ; 
       virtual unsigned                 getTargetNode() = 0 ; 
       virtual const char*              getName() const = 0 ; 
       virtual std::string              getSolidName(int mesh_id) = 0 ; 
       virtual int                      getLogicalVolumeIndex(int mesh_id) = 0 ; 
       virtual NParameters*             getCSGMetadata(int mesh_id) = 0 ;
       virtual NCSG*                    getCSG(int mesh_id) = 0 ; 

       virtual std::string              getMeshName(unsigned mesh_id) = 0 ;
       virtual unsigned                 getMeshNumPrimitives(unsigned mesh_id) = 0 ;
       virtual unsigned                 getNumInstances(unsigned mesh_idx) = 0 ;

       virtual bool                     isUsedGlobally(unsigned mesh_idx) = 0 ;
       virtual void                     setIsUsedGlobally(unsigned mesh_idx, bool iug) = 0 ;
       virtual const NSceneConfig*      getConfig() const = 0 ; 
       virtual const std::vector<unsigned>& getInstances(unsigned mesh_idx) = 0 ; // nodes that use the mesh
       virtual glm::mat4                getTransformMatrix( unsigned node_idx ) = 0 ; 


};







