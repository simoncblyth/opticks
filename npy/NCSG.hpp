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

#include <string>
#include <vector>
#include <map>

#include "NGLM.hpp"
#include "OpticksCSG.h"
#include "plog/Severity.h"

template <typename T> class NPY ; 
#include "NPY_API_EXPORT.hh"

/**
NCSG
======

Workflow 1 : NCSG::Load from python CSG defined geometry thru to transport buffers 
-------------------------------------------------------------------------------------

Steps 1,2,3 are done by NCSG::Load

0. serialize python defined geometry (see tboolean-box--) into src* buffers 
1. loadsrc the src* buffers into NCSGData.m_src* 
2. import the NCSGData.m_src* buffers into nnode tree
3. export the nnode tree into NCSGData transport buffers m_nodes, m_gtransforms, m_planes

Workflow 2 : from "g4code" generated G4VSolids thru to transport buffers 
---------------------------------------------------------------------------

Initial stages using the code generation not pinned down, but would like
to be similar to tboolean-/opticks-nnt examples, see y4csg/tests/Y4CSGTest.cc.
Revolves around the below conversions::
    
    G4VSolid* solid = ...
    nnode* root = X4Solid::Convert(solid);                  // direct tree conversion from G4VSolid -> nnode  (with g4code generation on side)  
    NCSG* csg = NCSG::Adopt( root ); // NCSG(nnode*) ctor, then export to transport buffers  

Note that cfg4/CMaker can do the opposite conversion, creating a G4VSolid from nnode



NCSG implementation
----------------------

cf dev/csg/csg.py 

* NB two constructors:

  1. deserialize from a python written treedir
  2. serialize an nnode tree

* hmm the raw loaded buffer lacks bounding boxes when derived from user input, 
  to get those need to vivify the CSG tree and the export it back to the buffer


TODO
-------

1. do not like the double ctors as it leads to schizophrenic class 
   
   * perhaps can start to tidy with an NCSGData constituent
     that holds the transport buffers

2. break up the monolith. But its a pivotal class, so rejigging 
   it will be time consuming. 

3. evolved like this because of too many usages
   including a input from CSG python


setIsUsedGlobally : always now true ?
-------------------------------------------

::

    epsilon:npy blyth$ opticks-find setIsUsedGlobally 
    ./npy/NGLTF.cpp:void NGLTF::setIsUsedGlobally(unsigned mesh_idx, bool iug)
    ./npy/NScene.cpp:    //if(n->repeatIdx == 0) setIsUsedGlobally(n->mesh, true );
    ./npy/NScene.cpp:    m_source->setIsUsedGlobally(n->mesh, true );
    ./npy/NCSG.cpp:void NCSG::setIsUsedGlobally(bool usedglobally )
    ./npy/NCSG.cpp:    tree->setIsUsedGlobally(true);
    ./npy/NCSG.hpp:        void setIsUsedGlobally(bool usedglobally);
    ./npy/NGeometry.hpp:       virtual void                     setIsUsedGlobally(unsigned mesh_idx, bool iug) = 0 ;
    ./npy/NGLTF.hpp:        void                         setIsUsedGlobally(unsigned mesh_idx, bool iug);



Where NCSG is used
-------------------

void GGeoTest::loadCSG(const char* csgpath, std::vector<GVolume*>& volumes)

    GGeoTest is the primary user of NCSG, with method GGeoTest::loadCSG
    invoking NCSG::Deserialize to create vectors of csg trees. 
    Each tree is converted into a GVolume by GMaker::makeFromCSG(NCSG* tree).  

    This python defined CSG node tree approach has replaced
    the old bash configured GGeoTest::createCsgInBox.


GVolume* GMaker::makeFromCSG(NCSG* csg)

    Coordinates:

    * NPolygonizer -> GMesh 
    * GParts::make, create analytic description
    * GVolume, container for GMesh and GParts

GParts* GParts::make( NCSG* tree)

    Little to do, just bndlib hookup :
    huh but thats done in GMaker::makeFromCSG too...

    * Is GParts is serving any purpose for NCSG ?
      YES : it provides merging of analytic volumes
      in parallel with the merging of triangulated meshes


**/

struct nvec4 ; 
union nquad ; 
struct nnode ; 
struct nmat4pair ; 
struct nmat4triple ; 
struct nbbox ; 
struct NSceneConfig ; 
struct NNodeNudger ; 

class BMeta ; 

class NCSGData ; 
class NPYMeta ; 
class NPYList ; 

class NNodePoints ; 
class NNodeUncoincide ; 
class NTrianglesNPY ;

class NPY_API NCSG {
        friend class  NCSGList ; 
        friend struct NCSGLoadTest ; 
        typedef std::map<std::string, nnode*> MSN ; 
    public:
        static const plog::Severity LEVEL ; 
        static const unsigned MAX_EXPORT_HEIGHT ;  
        static const float SURFACE_EPSILON ; 
        static NNodeNudger* MakeNudger(const char* msg, nnode* root, float surface_epsilon ); 

        static std::string TestVolumeName(const char* shapename, const char* suffix, int idx) ; 
        std::string getTestLVName() const ;
        std::string getTestPVName() const ;

        // cannot be "const nnode* root" because of the nudger
        // Adopt was formerly FromNode
        static NCSG* Adopt(nnode* root, const NSceneConfig* config, unsigned soIdx, unsigned lvIdx );
        static NCSG* Adopt(nnode* root, const char* config        , unsigned soIdx, unsigned lvIdx );
        static NCSG* Adopt(nnode* root);

        static void PrepTree(nnode* root);  


        static NCSG* Load(const char* treedir);
        static NCSG* Load(const char* treedir, const char* gltfconfig);
        static NCSG* Load(const char* treedir, const NSceneConfig* config );

        static BMeta* LoadMetadata( const char* treedir, int item=-1 );  // -1 for global


        NNodeUncoincide* make_uncoincide() const ;
        NNodeNudger*     get_nudger() const ;
        int              get_num_coincidence() const ;
        std::string      desc_coincidence() const ;

        void resizeToFit( const nbbox& container, float scale, float delta ) const ;
    public:
        NTrianglesNPY* polygonize();
        NTrianglesNPY* polygonize_bbox_placeholder();

        NTrianglesNPY* getTris() const ;

        NCSGData* getCSGData() const ;
        NPYList* getNPYList() const ;
        NPYMeta*  getMeta() const ;

    public:
        // passthru to root
        unsigned    get_type_mask() const ;
        unsigned    get_tree_mask() const ;
        unsigned    get_leaf_mask() const ;
        std::string get_type_mask_string() const ;
    public:
        nbbox     bbox() const ;
        glm::vec4 bbox_center_extent() const ;
        nbbox     bbox_surface_points() const ;
    public:
        const std::vector<glm::vec3>& getSurfacePoints() const ;
        unsigned getNumSurfacePoints() const ;
        float    getSurfaceEpsilon() const ; 
        static   glm::uvec4 collect_surface_points(std::vector<glm::vec3>& surface_points, const nnode* root, const NSceneConfig* config, unsigned verbosity, float epsilon );
    private:
        glm::uvec4 collect_surface_points();

    public:
        bool      has_placement_translation() const ; 
        glm::vec3 get_placement_translation() const ; 

        bool      has_placement_transform() const ; 
        glm::mat4 get_placement_transform() const ; 

        void      set_translation( float x, float y, float z); 
        void      set_centering(); 

        bool      has_root_transform() const ; 
        glm::mat4 get_root_transform() const ; 

    public:
        // kept in metadata : survives persisting 
        void        set_lvname(const char* name);
        void        set_soname(const char* name);
        void        set_balanced(bool balanced); 
        void        set_altindex(int altindex); 
        void        set_emit(int emit);   // used by --testauto 
        void        set_emitconfig(const char* emitconfig);  

        std::string get_lvname() const ;
        std::string get_soname() const ;
        bool        is_balanced() const ;
        int         get_altindex() const ;
        int         get_emit() const ;
        bool        is_emitter() const ;
        const char* get_emitconfig() const ;
    public:
        int         get_treeindex() const ;
        int         get_depth() const ;
        int         get_nchild() const ;
        bool        is_skip() const ;
        bool        is_uncoincide() const ;
    public:
        std::string meta() const ;
        std::string smry() const ;
    public:
        void dump(const char* msg="NCSG::dump") const ;
        void dump_surface_points(const char* msg="NCSG::dump_surface_points", unsigned dmax=20) const ;

        std::string desc();
        std::string brief() const ;
   public:
        void setIsUsedGlobally(bool usedglobally);
   public:
   public:
        const char*  getBoundary() const ;
        const char*  getTreeDir() const ;
        const char*  getTreeName() const ;
        int          getTreeNameIdx() const ;  // will usually be the lvidx

        unsigned     getIndex() const ;
        int          getVerbosity() const ;
    public:
        bool         isContainer() const ;
        bool         isContainerAutoSize() const ;
        float        getContainerScale() const ;

    public:
        bool         isUsedGlobally() const ;
    public:
        unsigned     getHeight() const ;

    public:
        // from NCSGData.m_csgdata, principal consumer is GParts::make
        unsigned     getNumNodes() const ;

        NPY<float>*    getNodeBuffer() const ;
        NPY<float>*    getTransformBuffer() const ;
        NPY<float>*    getGTransformBuffer() const ;
        NPY<float>*    getPlaneBuffer() const ;
        NPY<unsigned>* getIdxBuffer() const ;
    public:
        // from NPYMeta.m_meta 
        BMeta*         getMeta(int idx) const ;

    public:
        nnode*       getRoot() const ;
        OpticksCSG_t getRootType() const ;  
        const char*  getRootCSGName() const ;
        bool         isBox() const ; 
        bool         isBox3() const ; 


        unsigned getNumTriangles() const ;
    public:
        void check() const ;
    private:
        void checkroot() const ;
        void check_r(const nnode* node) const ; 
        void check_node(const nnode* node ) const ;
    public:
        void setIndex(unsigned index);
        void setVerbosity(int verbosity);
    private:
        NCSG(const char* treedir); // Deserialize
        NCSG( nnode* root);  // Serialize : cannot be const because of the nudger
    public:
        void setBoundary(const char* boundary); // for --testauto
    private:
        void init(); 
    private:
        // Deserialize branch 
        void setConfig(const NSceneConfig* config);
        const NSceneConfig* getConfig() const ;   

    public:
        void savesrc(const char* idpath, const char* rela, const char* relb ) const ; 
        void savesrc(const char* treedir) const ;
    private:
        void loadsrc();
        void postload();
        void increaseVerbosity(int verbosity);

   private:
        // import src buffers of nodes/transforms into a node tree 
        void import();
        void import_leaf();
        void import_tree();
        void import_tree_list_(unsigned idx0); 
        void import_tree_list_subs_( nnode* n, unsigned sub_num, unsigned idx0 ); 

        void import_list();
        nnode* import_list_node( unsigned idx ); 

        void postimport();
        void postimport_uncoincide();
        void postimport_autoscan();

        nnode* import_tree_r(unsigned idx, nnode* parent=NULL);
        nnode* import_tree_operator( unsigned idx, OpticksCSG_t typecode );
        nnode* import_primitive( unsigned idx, OpticksCSG_t typecode );
        void import_srcplanes(nnode* node);
        void import_srcvertsfaces(nnode* node);

    public:
        void postchange(); 
        // collect global transforms into m_gtransforms and sets the node->gtransform and node->gtransform_idx refs
        void collect_global_transforms() ;
    private:
        void collect_global_transforms_list(nnode* node); 
        void collect_global_transforms_r(nnode* node) ;
        void collect_global_transforms_node(nnode* node);
        unsigned addUniqueTransform( const nmat4triple* gtransform );

    private:
        // Serialize branch 
        // export node tree into a buffers (complete binary tree buffer of nodes, transforms, planes)
        void export_();
        void export_idx();
        void export_srcidx();
        void export_tree_();
        void export_tree_list_prepare_(unsigned num_tree_nodes_); 
        void export_tree_list_(unsigned num_tree_nodes_); 
        void export_tree_list_subs_( const nnode* n, unsigned sub_num, unsigned idx0 ); 


        void export_tree_r(nnode* node, unsigned idx);
        void export_list_();
        void export_leaf_();
        static void check_subs(const nnode* node) ; 


        // collects gtransform into the tran buffer and sets gtransform_idx 
        // into the node tree needed to prepare a G4 directly converted solid to go to GPU 
        void export_node( nnode* node, unsigned idx );
        void export_srcnode( nnode* node, unsigned idx);
        void export_planes( nnode* node, NPY<float>* _dest );
        void export_itransform( nnode* node, NPY<float>* _dest ); 
        // exporting writes things like indices into the node, so cannot be const 
    public:
        void setSOIdx(unsigned soIdx);
        void setLVIdx(unsigned lvIdx);
        unsigned getSOIdx() const ; 
        unsigned getLVIdx() const ; 
    public:
        bool isProxy() const ; 
        unsigned getProxyLV() const ;


    public:
        void setOther(NCSG* other); 
        NCSG* getOther() const ;   
    private:
        NPYMeta*         m_meta ; 
        const char*      m_treedir ; 
        unsigned         m_index ; 
        float            m_surface_epsilon ; 
        int              m_verbosity ;  
        bool             m_usedglobally ; 
        bool             m_balanced ; 
        nnode*           m_root ;  
        NNodePoints*     m_points ; 
        NNodeUncoincide* m_uncoincide ; 
        NNodeNudger*     m_nudger ; 

        NCSGData*        m_csgdata ; 

        bool             m_adopted ; 
        const char*         m_boundary ; 
        const NSceneConfig* m_config ; 

        glm::vec3   m_gpuoffset ; 
        int         m_proxylv ;  
        int         m_container ;  
        float       m_containerscale ;  
        int         m_containerautosize ;  
        
        NTrianglesNPY*         m_tris ; 
        std::vector<glm::vec3> m_surface_points ; 

        unsigned    m_soIdx ;   // debugging 
        unsigned    m_lvIdx ;   // debugging 

        NCSG*       m_other ; 
        int         m_height ;   // -1: when root is not a binary tree (eg for multiunions), otherwise tree height 

};


