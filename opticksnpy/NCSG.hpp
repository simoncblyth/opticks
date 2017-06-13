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

cf dev/csg/csg.py 

* this can currently only Deserialize from a python written directory 

* hmm the raw loaded buffer lacks bounding boxes when derived from user input, 
  to get those need to vivify the CSG tree and the export it back to the buffer


Where NCSG is used
-------------------

void GGeoTest::loadCSG(const char* csgpath, std::vector<GSolid*>& solids)

    GGeoTest is the primary user of NCSG, with method GGeoTest::loadCSG
    invoking NCSG::Deserialize to create vectors of csg trees. 
    Each tree is converted into a GSolid by GMaker::makeFromCSG(NCSG* tree).  

    This python defined CSG node tree approach has replaced
    the old bash configured GGeoTest::createCsgInBox.


GSolid* GMaker::makeFromCSG(NCSG* csg)

    Coordinates:

    * NPolygonizer -> GMesh 
    * GParts::make, create analytic description
    * GSolid, container for GMesh and GParts

GParts* GParts::make( NCSG* tree)

    Little to do, just bndlib hookup :
    huh but thats done in GMaker::makeFromCSG too...

    * TODO: review if GParts is serving any purpose for NCSG


**/

struct nvec4 ; 
union nquad ; 
struct nnode ; 
struct nmat4pair ; 
struct nmat4triple ; 

class NParameters ; 
class NTrianglesNPY ;

class NPY_API NCSG {
        friend struct NCSGLoadTest ; 
        typedef std::map<std::string, nnode*> MSN ; 
    public:
        enum { NJ = 4, NK = 4, MAX_HEIGHT = 10 };
        static const char* FILENAME ; 
        static const unsigned NTRAN ; 

        static unsigned NumNodes(unsigned height);
        static bool Exists(const char* base);
        static int Deserialize(const char* base, std::vector<NCSG*>& trees, int verbosity );
        static int Polygonize( const char* base, std::vector<NCSG*>& trees, int verbosity );
        static NCSG* FromNode(nnode* root, const char* boundary);
        static NCSG* LoadTree(const char* treedir, bool usedglobally=false, int verbosity=0, bool polygonize=false );
        static NParameters* LoadMetadata(const char* treedir);
    public:
        NTrianglesNPY* polygonize();
        NTrianglesNPY* getTris();
    public:
        template<typename T> T getMeta(const char* key, const char* fallback );
        template<typename T> void setMeta(const char* key, T value);
        std::string lvname();
        std::string pvname();
        std::string soname();
        int treeindex();
        int depth();
        int nchild();
        bool isSkip();
        std::string meta();
        std::string smry();
    public:
        void dump(const char* msg="NCSG::dump");
        std::string desc();
   public:
        void setIsUsedGlobally(bool usedglobally);
   public:
        const char*  getTreeDir();
        unsigned     getIndex();
        int          getVerbosity();
        bool         isUsedGlobally();
        const char*  getBoundary();
        NPY<float>*  getNodeBuffer();
        NPY<float>*  getTransformBuffer();
        NPY<float>*  getGTransformBuffer();
        NPY<float>*  getPlaneBuffer();
        NParameters* getMetaParameters();
        unsigned     getNumNodes();
        unsigned     getNumTransforms();
        unsigned     getHeight();
        nnode*       getRoot();
        OpticksCSG_t getRootType();  
    public:
        float getContainerScale();
        bool  isContainer();
        unsigned getNumTriangles();
    public:
        void check();
        void check_r(nnode* node); 
        void setIndex(unsigned index);
        void setVerbosity(int verbosity);
    private:
        // Deserialize
        NCSG(const char* treedir);
         // Serialize 
        NCSG(nnode* root);
    private:
        // Deserialize branch 
        void setBoundary(const char* boundary);
        unsigned getTypeCode(unsigned idx);
        unsigned getTransformIndex(unsigned idx);
        bool     isComplement(unsigned idx);
        nquad getQuad(unsigned idx, unsigned j);
    private:
        void load();
        void loadMetadata();
        void loadNodes();
        void loadTransforms();
        void loadPlanes();
    private:
        void import();
        nnode* import_r(unsigned idx, nnode* parent=NULL);
        nnode* import_primitive( unsigned idx, OpticksCSG_t typecode );
        nnode* import_operator( unsigned idx, OpticksCSG_t typecode );
        void import_planes(nnode* node);
    private:
        nmat4pair*   import_transform_pair(unsigned itra);
        nmat4triple* import_transform_triple(unsigned itra);
        unsigned addUniqueTransform( nmat4triple* gtransform );
    private:
         // Serialize branch
        void export_r(nnode* node, unsigned idx);
        void export_();
    private:
        unsigned    m_index ; 
        int         m_verbosity ;  
        bool        m_usedglobally ; 
        nnode*      m_root ;  
        const char* m_treedir ; 

        NPY<float>* m_nodes ; 
        NPY<float>* m_transforms ; 
        NPY<float>* m_gtransforms ; 
        NPY<float>* m_planes ;

        NParameters* m_meta ; 
        unsigned    m_num_nodes ; 
        unsigned    m_num_transforms ; 
        unsigned    m_num_planes ;
 
        unsigned    m_height ; 
        const char* m_boundary ; 
        glm::vec3   m_gpuoffset ; 
        int         m_container ;  
        float       m_containerscale ;  

         NTrianglesNPY* m_tris ; 


};


