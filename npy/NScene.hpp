/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once
#include "NPY_API_EXPORT.hh"

#include "NGLM.hpp"
#include <map>
#include <vector>
#include <string>

class NCSG ; 
class BTxt ; 

#ifdef OLD_PARAMETERS
class X_BParameters ; 
#else
class NMeta ; 
#endif

class NGeometry ;
 
struct nd ; 
struct nbbox ; 
struct nnode ; 
struct nmat4triple ; 
struct NSceneConfig ; 

template<class T> class NPY ;
template<class T> class Counts ;

#include "NSceneConfig.hpp"  // for enums 
class NGLTF ; 


/*

NScene
===============

* NGLTF was formerly the base class, now moved to constituent m_ngltf
  in order to distance NScene from GLTF mechanics and make
  it work with nd/NCSG coming from other sources like an X4 provider

  * in order to do this are constraining the interface with NGeometry protocol  

* NScene does far too much for one class.

  * pulls in NCSG extras from sc.py 
  * finds repeat geometry instances 

* NScene is consumed directly only by GScene, 

* NScene::Load from GScene::GScene 


Scene files in glTF format are created by opticks/analytic/sc.py 
which parses the input GDML geometry file and writes the mesh (ie solid 
shapes) as np ncsg and the tree structure as json/gltf.

NScene imports the gltf using its NGLTF based (YoctoGL external)
creating a nd tree. The small CSG node trees for each solid
are polygonized on load in NScene::load_mesh_extras.

*/



class NPY_API NScene 
{
    public:
        static NScene* Load( const char* gltfbase, const char* gltfname, const char* idfold, NSceneConfig* gltfconfig, int dbgnode, int scene_idx=0  ) ;

        NScene(NGeometry* source, const char* idfold, int dbgnode );
        //NScene(const char* base, const char* name, const char* idfold, NSceneConfig* config, int dbgnode, int scene_idx=0  );

        nd*      getRoot() const ;
        unsigned getNumMeshes() const ;

        NCSG*    getCSG(unsigned mesh_idx) const ;
        NCSG*    findCSG(const char* soname, bool startswith) const ;

        void dumpNd(unsigned idx, const char* msg="NScene::dumpNd");
        void dumpNdTree(const char* msg="NScene::dumpNdTree");
        void dumpCSG(const char* dbgmesh=NULL, const char* msg="NScene::dumpCSG") const  ; 

        unsigned getVerbosity();
        unsigned getTargetNode();
        const NSceneConfig* getConfig();
    public: 
         // from gltfconfig
         NSceneConfigBBoxType bbox_type() const ; 
         const char* bbox_type_string() const ;
    public:
        // TODO: should be accessible from NCSG 
        int  lvidx(unsigned mesh_id) const ;
        void collect_mesh_nodes(std::vector<unsigned>& nodes, unsigned mesh) const ;
        std::string present_mesh_nodes(std::vector<unsigned>& nodes, unsigned dmax=10) const ;
        std::string desc() const ;
    private:
        void collect_mesh_nodes_r(nd* n, std::vector<unsigned>& nodes, unsigned mesh) const ;
    private:
        void init();
        void import();
        void init_lvlists();
        void write_lvlists();
    private:
        template<typename T> T getCSGMeta(unsigned mesh_id, const char* key, const char* fallback ) const ;
        std::string lvname(unsigned mesh_id) const ;
        std::string soname(unsigned mesh_id) const ;
        int         height(unsigned mesh_id) const ;
        int         nchild(unsigned mesh_id) const ;
        bool        isSkip(unsigned mesh_id) const ;
        std::string meshmeta(unsigned mesh_id) const ;
    private:
        void load_asset_extras();
        void load_csg_metadata();
        void load_mesh_extras();
        void postimportmesh();
        void find_repeat_candidates();
    public:
        bool operator()(const std::string& pdig);
        bool is_contained_repeat( const std::string& pdig, unsigned levels ) ;
        void dump_repeat_candidate(unsigned i) const ;
        void dump_repeat_candidates() const;
    private:
        nd*  import_r(int idx, nd* parent, int depth); // recursive translation of ygltf node tree into nd tree
        void postimportnd();

        void dumpNdTree_r(nd* n);
        void count_progeny_digests();
        void count_progeny_digests_r(nd* n);
        void markGloballyUsedMeshes_r(nd* n);
    private:
        unsigned deviseRepeatIndex(const std::string& pdig);
        unsigned deviseRepeatIndex_0(nd* n);
        void labelTree();

#ifdef OLD_LABEL_TREE
        void labelTree_r(nd* n, unsigned /*ridx*/);
#else
        void labelTree_r(nd* n, unsigned ridx);
#endif

    public:
        void     dumpRepeatCount();
        unsigned getRepeatCount(unsigned ridx);
        unsigned getNumRepeats();
        NPY<float>*  makeInstanceTransformsBuffer(unsigned mesh_idx);

    private:
        bool is_dbgnode( const nd* n) const ;
    private:
        // cross structural node geometry checking 
        void check_surf_containment() ; 
        void check_surf_containment_r(const nd* node) ; 
        glm::uvec4 check_surf_points( const nd* n ) const ;
        void debug_node(const nd* node) const ; 

    private:
        void update_aabb();
        void update_aabb_r(nd* node);
        void check_aabb_containment() ; 
        void check_aabb_containment_r(const nd* node) ; 
        nbbox calc_aabb(const nd* node, bool global) const ;
    private:
        nnode*    getSolidRoot(const nd* n) const ;
        float    sdf_procedural( const nd* n, const glm::vec3& q_) const  ; 

    private:
        NGeometry*                        m_source ; 
    private:

        nd*                               m_root ; 

#ifdef OLD_PARAMETERS
        std::map<unsigned, X_BParameters*>  m_csg_metadata ;
#else
        std::map<unsigned, NMeta*>       m_csg_metadata ;
#endif


        std::map<unsigned, NCSG*>         m_csg ; 
        std::map<unsigned, int>           m_csg_lvIdx ; 
       
        std::map<unsigned, unsigned>      m_mesh2ridx ;
        std::map<unsigned, unsigned>      m_repeat_count ;


        unsigned                          m_num_nodes ; 
        const char*                       m_idfold ; 
        const NSceneConfig*               m_config ; 
        int                               m_dbgnode ; 
        unsigned                          m_containment_err ; 
        unsigned                          m_verbosity ; 
        unsigned                          m_targetnode ; 
        unsigned                          m_num_global ; 
        unsigned                          m_num_csgskip ; 
        unsigned                          m_num_placeholder ; 
        BTxt*                             m_csgskip_lvlist ; 
        BTxt*                             m_placeholder_lvlist ; 
        unsigned                          m_node_count ; 
        unsigned                          m_label_count ; 
        Counts<unsigned>*                 m_digest_count ;

        std::vector<std::string>          m_repeat_candidates ;
        std::vector<unsigned>             m_dbgnode_list ;
        glm::uvec4                        m_surferr ;  

        bool                              m_triple_debug ; 
};

