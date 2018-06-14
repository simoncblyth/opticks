#pragma once

#include <array>
#include <string>
#include <map>
#include <vector>
#include <glm/fwd.hpp>

#include "NPY_API_EXPORT.hh"

template<class T> class NPY ;

struct nd ; 

namespace ygltf 
{
    struct glTF_t ;  
    struct fl_gltf ;
    struct fl_mesh ;
    struct node_t ;
    struct mesh_t ;
}

struct NSceneConfig ; 

/**
NGLTF
======

Based on ygltf parser from YoctoGL external, loads GLTF file 
and provides simple interface to its content : transforms, 
mesh indices. 

NGLTF is the base class of NScene.

**/

class NPY_API NGLTF {
    public:
        NGLTF(const char* base, const char* name, const NSceneConfig* config, unsigned scene_idx);
    public:
        const char*         getBase() const ; 
        const char*         getName() const ; 
        const NSceneConfig* getConfig() const ;
    private:
        // init
        void load();     // loads m_gltf from file and flattens giving m_fgltf  
        void collect();  // multiply out the node transforms to give global transforms for each node in m_xf
    public:
        void dump(const char* msg="NGLTF::dump");
        void dump_scenes(const char* msg="NGLTF::dump_scenes");
        void dump_flat_nodes(const char* msg="NGLTF::dump_flat_nodes");
        void dump_mesh_totals(const char* msg="NGLTF::dump_mesh_totals", unsigned maxdump=10u );
        void dump_node_transforms(const char* msg="NGLTF::dump_node_transforms");
    public:
        NPY<float>*                  makeInstanceTransformsBuffer(unsigned mesh_idx);
        unsigned                     getNumMeshes() const ;
        unsigned                     getNumInstances(unsigned mesh_idx);
        const std::vector<unsigned>& getInstances(unsigned mesh_idx); // nodes that use the mesh
    public:
        // meshes that are used globally need to have gtransform slots for all primitives
        bool                         isUsedGlobally(unsigned mesh_idx);
        void                         setIsUsedGlobally(unsigned mesh_idx, bool iug);
    public:
        ygltf::mesh_t*               getMesh(unsigned mesh_id); 
        std::string                  getMeshName(unsigned mesh_id);
        unsigned                     getMeshNumPrimitives(unsigned mesh_id);
        template<typename T> T       getMeshExtras(int mesh_id, const char* key)  ;
        std::string                  getCSGPath(int mesh_id); 


        ygltf::fl_mesh*              getFlatNode(unsigned node_idx );
        const std::array<float, 16>& getFlatTransform(unsigned node_idx );
        const std::array<float, 16>& getNodeTransform(unsigned node_idx );
        glm::mat4                    getTransformMatrix(unsigned node_idx );
    public:
        void dumpAll();
        void dumpAllInstances( unsigned mesh_idx, unsigned maxdump=10u );
    public:
        unsigned                     getNumNodes() const  ;
        ygltf::node_t*               getNode(unsigned node_idx ) const ;
        const std::vector<int>&      getNodeChildren(unsigned node_idx) const  ; 
        nd*                          createNdConverted(unsigned node_idx, unsigned depth, nd* parent) const  ;
    public:
        void                         compare_trees_r(int idx); // recursive compare the ynode and nd trees
    public:
        std::string                  descFlatNode( unsigned node_idx );
        std::string                  descNode( unsigned node_idx );
        std::string                  desc() const ;

        template<typename T> T       getAssetExtras(const char* key)  ;

    protected:
        const char*           m_base ; 
        const char*           m_name ; 
        const NSceneConfig*   m_config ; 
        int                   m_scene_idx ; 
        ygltf::glTF_t*        m_gltf ;  
        ygltf::fl_gltf*       m_fgltf ;

        std::map<unsigned, std::vector<unsigned>>  m_mesh_instances ; 
        std::map<unsigned, unsigned>               m_mesh_totals ; 
        std::map<unsigned, bool>                   m_mesh_used_globally ; 
        std::map<unsigned, std::array<float,16>>   m_xf  ; 
        std::map<unsigned, unsigned>               m_node2traversal  ; 

};
