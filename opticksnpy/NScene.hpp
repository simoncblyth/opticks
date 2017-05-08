#pragma once
#include "NPY_API_EXPORT.hh"
#include <string>
#include <map>
#include <vector>
#include <glm/fwd.hpp>

NPY_API static std::string xform_string( const std::array<float, 16>& xform );


namespace ygltf 
{
    struct glTF_t ;  
    struct fl_gltf ;
    struct fl_mesh ;
    struct node_t ;
}

class NCSG ; 

template<class T> class NPY ;

struct nd ; 


class NPY_API NScene 
{
    public:
        NScene(const char* base, const char* name, int scene_idx=0  );
    public:
        void dump(const char* msg="NScene::dump");
        void dump_scenes(const char* msg="NScene::dump_scenes");
        void dump_flat_nodes(const char* msg="NScene::dump_flat_nodes");
    private:
        void load();
        void collect_mesh_totals(int scn_id=0);
        void collect_mesh_instances();
        void collect_node_transforms();
        void load_mesh_extras();
    private:
        void import();
        nd*  import_r(int idx, nd* parent, int depth);
        void dumpNdTree(const char* msg="NScene::dumpNdTree");
        void dumpNdTree_r(nd* n);
        nd*  getNd(int idx);
        void compare_trees();
        void compare_trees_r(int idx);
        const std::array<float,16>& getTransformCheck(unsigned node_idx);
    public:
        void dump_mesh_totals(const char* msg="NScene::dump_mesh_totals");
        void dump_node_transforms(const char* msg="NScene::dump_node_transforms");
    public:
        NPY<float>* makeInstanceTransformsBuffer(unsigned mesh_idx);
        unsigned getNumMeshes();
        unsigned getNumInstances(unsigned mesh_idx);
        int getInstanceNodeIndex( unsigned mesh_idx, unsigned instance_idx);
    private:
        ygltf::fl_mesh* getFlatNode(unsigned node_idx );
        const std::array<float, 16>& getFlatTransform(unsigned node_idx );
        const std::array<float, 16>& getNodeTransform(unsigned node_idx );
        glm::mat4 getTransformMatrix(unsigned node_idx );
    public:
        void dumpAll();
        void dumpAllInstances( unsigned mesh_idx);
        std::string descInstance( unsigned mesh_idx, unsigned instance_idx );
    public:
        ygltf::node_t*  getNode(unsigned node_idx );
        std::string     descFlatNode( unsigned node_idx );
        std::string     descNode( unsigned node_idx );
    private:
        const char*     m_base ; 
        const char*     m_name ; 
        int             m_scene_idx ; 
        ygltf::glTF_t*  m_gltf ;  
        ygltf::fl_gltf* m_fgltf ;
        nd*             m_root ; 

        typedef std::map<int, std::vector<int>>  MMI ; 
        MMI m_mesh_instances ; 

        std::map<int, int> m_mesh_totals ; 
        std::vector<NCSG*> m_csg_trees ; 

        std::map<unsigned, nd*> m_nd ; 
        std::map<unsigned, std::array<float,16>> m_xf  ; 
        std::map<unsigned, unsigned> m_node2traversal  ; 


};

