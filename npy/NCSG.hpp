#pragma once

#include <string>
#include <vector>
#include <map>

#include "NGLM.hpp"
#include "OpticksCSG.h"

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

#ifdef OLD_PARAMETERS
class X_BParameters ; 
#else
class NMeta ; 
#endif

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
        static const float SURFACE_EPSILON ; 

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

#ifdef OLD_PARAMETERS
        static X_BParameters* LoadMetadata( const char* treedir, int item=-1 );  // -1 for global
#else
        static NMeta* LoadMetadata( const char* treedir, int item=-1 );  // -1 for global
#endif


        NNodeUncoincide* make_uncoincide() const ;
        NNodeNudger*     make_nudger(const char* msg) const ;
        NNodeNudger*     get_nudger() const ;
        unsigned         get_num_coincidence() const ;
        std::string      desc_coincidence() const ;

        void adjustToFit( const nbbox& container, float scale, float delta ) const ;
    public:
        NTrianglesNPY* polygonize();
        NTrianglesNPY* getTris() const ;

        NCSGData* getCSGData() const ;
        NPYList* getNPYList() const ;
        NPYMeta*  getMeta() const ;

    public:
        // passthru to root
        unsigned    get_type_mask() const ;
        unsigned    get_oper_mask() const ;
        unsigned    get_prim_mask() const ;
        std::string get_type_mask_string() const ;
    public:
        nbbox bbox_analytic() const ;
        nbbox bbox_surface_points() const ;

        const std::vector<glm::vec3>& getSurfacePoints() const ;
        unsigned getNumSurfacePoints() const ;
        float    getSurfaceEpsilon() const ; 
        static   glm::uvec4 collect_surface_points(std::vector<glm::vec3>& surface_points, const nnode* root, const NSceneConfig* config, unsigned verbosity, float epsilon );
    private:
        glm::uvec4 collect_surface_points();

    public:
        void set_lvname(const char* name);
        void set_soname(const char* name);
    public:
        std::string lvname() const ;
        std::string soname() const ;
        int treeindex() const ;
        int depth() const ;
        int nchild() const ;
        bool isSkip() const ;
        bool is_uncoincide() const ;
        std::string meta() const ;
        std::string smry() const ;
    public:
        // used by --testauto 
        void setEmit(int emit);  
        void setEmitConfig(const char* emitconfig);  
    public:
        bool        isEmit() const ;
        int         getEmit() const ;
        const char* getEmitConfig() const ;
    public:
        void dump(const char* msg="NCSG::dump");
        void dump_surface_points(const char* msg="NCSG::dump_surface_points", unsigned dmax=20) const ;

        std::string desc();
        std::string brief() const ;
   public:
        void setIsUsedGlobally(bool usedglobally);
   public:
        const char*  getBoundary() const ;
        const char*  getTreeDir() const ;
        const char*  getTreeName() const ;
        int          getTreeNameIdx() const ;  // will usually be the lvidx

        unsigned     getIndex() const ;
        int          getVerbosity() const ;
    public:
        bool         isContainer() const ;
        float        getContainerScale() const ;
        bool         isUsedGlobally() const ;



    public:
        // from NCSGData.m_csgdata, principal consumer is GParts::make

        unsigned     getHeight() const ;
        unsigned     getNumNodes() const ;

        NPY<float>*    getNodeBuffer() const ;
        NPY<float>*    getTransformBuffer() const ;
        NPY<float>*    getGTransformBuffer() const ;
        NPY<float>*    getPlaneBuffer() const ;
        NPY<unsigned>* getIdxBuffer() const ;
    private: 
        /*
        NPY<float>*    getSrcNodeBuffer() const ;
        NPY<unsigned>* getSrcIdxBuffer() const ;
        NPY<float>*    getSrcPlaneBuffer() const ;
        NPY<float>*    getSrcTransformBuffer() const ;
        */
    public:
        // from NPYMeta.m_meta 

#ifdef OLD_PARAMETERS
        X_BParameters*   getMeta(int idx) const ;
#else
        NMeta*         getMeta(int idx) const ;
#endif


        nnode*       getRoot() const ;
        OpticksCSG_t getRootType() const ;  
        unsigned getNumTriangles() const ;
    public:
        void check() const ;
    private:
        void check_r(const nnode* node) const ; 
        void check_node(const nnode* node ) const ;
    public:
        void setIndex(unsigned index);
        void setVerbosity(int verbosity);
    private:
        // Deserialize
        NCSG(const char* treedir);

        // Serialize 
        NCSG( nnode* root);  // cannot be const because of the nudger
    public:
        // for --testauto
        void setBoundary(const char* boundary);
    private:
        // Deserialize branch 
        void setConfig(const NSceneConfig* config);

    public:
        void savesrc(const char* treedir) const ;
    private:
        void loadsrc();
        void postload();
        void increaseVerbosity(int verbosity);

   private:
        // import src buffers of nodes/transforms into a node tree 
        void import();
        void postimport();
        void postimport_uncoincide();
        void postimport_autoscan();

        nnode* import_r(unsigned idx, nnode* parent=NULL);
        nnode* import_primitive( unsigned idx, OpticksCSG_t typecode );
        nnode* import_operator( unsigned idx, OpticksCSG_t typecode );
        void import_srcplanes(nnode* node);
        void import_srcvertsfaces(nnode* node);

    public:
        // collect global transforms into m_gtransforms and sets the node->gtransform and node->gtransform_idx refs
        void collect_global_transforms() ;
    private:
        void collect_global_transforms_r(nnode* node) ;
        void collect_global_transforms_visit(nnode* node);
        unsigned addUniqueTransform( const nmat4triple* gtransform );

    private:
        // Serialize branch 
        // export node tree into a buffers (complete binary tree buffer of nodes, transforms, planes)
        void export_r(nnode* node, unsigned idx);
        void export_idx();
        void export_();

        // collects gtransform into the tran buffer and sets gtransform_idx 
        // into the node tree needed to prepare a G4 directly converted solid to go to GPU 
        void export_node( nnode* node, unsigned idx );
        //void export_gtransform(nnode* node);
        void export_srcnode(nnode* node, unsigned idx);
        void export_planes(nnode* node, NPY<float>* _dest );
        void export_itransform( nnode* node, NPY<float>* _dest ); 
    public:
        void setSOIdx(unsigned soIdx);
        void setLVIdx(unsigned lvIdx);
        unsigned getSOIdx() const ; 
        unsigned getLVIdx() const ; 

    private:
        const char*      m_treedir ; 
        unsigned         m_index ; 
        float            m_surface_epsilon ; 
        int              m_verbosity ;  
        bool             m_usedglobally ; 
        nnode*           m_root ;  
        NNodePoints*     m_points ; 
        NNodeUncoincide* m_uncoincide ; 
        NNodeNudger*     m_nudger ; 

        NCSGData*        m_csgdata ; 
        NPYMeta*         m_meta ; 

        bool             m_adopted ; 
        const char*         m_boundary ; 
        const NSceneConfig* m_config ; 

        glm::vec3   m_gpuoffset ; 
        int         m_container ;  
        float       m_containerscale ;  
        
        NTrianglesNPY*         m_tris ; 
        std::vector<glm::vec3> m_surface_points ; 

        unsigned    m_soIdx ;   // debugging 
        unsigned    m_lvIdx ;   // debugging 



};


