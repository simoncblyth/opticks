#pragma once

#include <map>
#include <string>
#include <vector>

#include <glm/fwd.hpp>
#include "OpticksCSG.h"

struct npart ; 
struct NSlice ; 
template <typename T> class NPY ;
template <typename T> class GMatrix ;

#include "NCSG.hpp"

//class NCSG ; 

//struct guint4 ; 
struct nivec4 ; 

struct gbbox ; 
struct gfloat3 ; 


class GItemList ; 
class GBndLib ; 

/**
GParts
======= 

Creates *primitive* buffer (formerly called *solid*) from the *parts* buffer
the *parts* buffer .npy for DYB PMT geometry is created by detdesc partitioning with pmt-/tree.py 
OR for test geometries it is created part-by-part using methods of the npy primitive structs, see eg::

   npy/NPart.hpp
   npy/NSphere.hpp 


GParts holds boundary specifications as lists of strings
that are only converted into actual boundaries with indices pointing 
at materials and surface by GParts::registerBoundaries which 
is invoked by GParts::close which happens late 
(typically within oxrap just before upload to GPU). 

This approach was adopted to allow dynamic addition of geometry and
boundaries, which is convenient for testing.

Lifecycle
-----------

Single Tree GParts created from from NCSG by GScene
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GParts are created from the NCSG in GScene::createVolume where they get attached to a GVolume::

    629 GVolume* GScene::createVolume(nd* n, unsigned depth, bool& recursive_select  ) // compare with AssimpGGeo::convertStructureVisit
    630 {
    ...
    644     NCSG*   csg =  getCSG(rel_mesh_idx);

    661     std::string bndspec = lookupBoundarySpec(solid, n);  // using just transferred boundary from tri branch
    662 
    663     GParts* pts = GParts::make( csg, bndspec.c_str(), m_verbosity  ); // amplification from mesh level to node level 
    664 
    665     pts->setBndLib(m_tri_bndlib);
    666 
    667     solid->setParts( pts );


Merged GParts are born with GMergedMesh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     45 GMergedMesh::GMergedMesh(unsigned int index)
     46        :
     47        GMesh(index, NULL, 0, NULL, 0, NULL, NULL),
     48        m_cur_vertices(0),
     49        m_cur_faces(0),
     50        m_cur_solid(0),
     51        m_num_csgskip(0),
     52        m_cur_base(NULL),
     53        m_parts(new GParts())
     54 {
     55 }


GParts merging happens via GMergedMesh::mergeVolumeAnalytic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    525 void GMergedMesh::mergeVolumeAnalytic( GParts* pts, GMatrixF* transform, unsigned verbosity )
    526 {
    527     // analytic CSG combined at node level  
    ...
    535     if(transform && !transform->isIdentity())
    536     {
    537         pts->applyPlacementTransform(transform, verbosity );
    538     }
    539     m_parts->add(pts, verbosity);
    540 }


Primary GParts usage by OGeo::makeAnalyticGeometry feeding cu/intersect_analytic.cu
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    470 optix::Geometry OGeo::makeAnalyticGeometry(GMergedMesh* mm)
    471 {
    ...
    480     GParts* pts = mm->getParts(); assert(pts && "GMergedMesh with GEOCODE_ANALYTIC must have associated GParts, see GGeo::modifyGeometry ");
    481 
    482     if(pts->getPrimBuffer() == NULL)
    483     {
    485         pts->close();
    487     } 
    ... 
    496     NPY<float>*     partBuf = pts->getPartBuffer(); assert(partBuf && partBuf->hasShape(-1,4,4));    // node buffer
    497     NPY<float>*     tranBuf = pts->getTranBuffer(); assert(tranBuf && tranBuf->hasShape(-1,3,4,4));  // transform triples (t,v,q) 
    498     NPY<float>*     planBuf = pts->getPlanBuffer(); assert(planBuf && planBuf->hasShape(-1,4));      // planes used for convex polyhedra such as trapezoid
    499     NPY<int>*       primBuf = pts->getPrimBuffer(); assert(primBuf && primBuf->hasShape(-1,4));      // prim
    500     NPY<unsigned>*  idBuf = mm->getAnalyticInstancedIdentityBuffer(); assert(idBuf && ( idBuf->hasShape(-1,4) || idBuf->hasShape(-1,1,4)));
    501      // PmtInBox yielding -1,1,4 ?
    502 



Mesh-type or node-type
~~~~~~~~~~~~~~~~~~~~~~~~~

Hmm boundary spec is a node-type-qty, not a mesh-type-qty 
so it does not belong inside GParts (a mesh-type-qty)
... there are relatively few mesh-type-qty for 
each distinct shape (~249), but much more node-type-qty (~12k)
 
BUT: GParts combination means that it kinda transitions 
between mesh-type when just for a single unplaced shape
into node-type once applyPlacementTransform is used by GMergedMesh::mergeVolumeAnalytic


persisted structure detailed in GParts.rst
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* for examination of structure of multi-complete tree buffers see GParts.rst


**/

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GParts { 
       // users of setPrimFlag
       friend class GGeoTest ;
       friend class GPmt ; 
    public:
       // conventional names for interfacing
       static const char* CONTAINING_MATERIAL ; 
       static const char* SENSOR_SURFACE ; 
       static const int NTRAN ; 


       static void BufferTags(std::vector<std::string>& tags)  ;
       static const char* BufferName(const char* tag) ;
       static NPY<float>* LoadBuffer(const char* dir, const char* tag);

    public:
        // buffer layout, must match locations in pmt-/tree.py:convert 
        enum { 
              QUADS_PER_ITEM = 4, 
              NJ = 4,
              NK = 4,
              SK = 4  
            } ;
    public:
      //
        static GParts* make(const npart& pt, const char* spec);
        static GParts* make(OpticksCSG_t csgflag, glm::vec4& param, const char* spec);
        static GParts* make(const NCSG* tree, const char* spec, unsigned verbosity );
    public:
        static GParts* combine(std::vector<GParts*> subs, unsigned verbosity);
        static GParts* combine(GParts* onesub,            unsigned verbosity );   // for consistent handling between 1 and many 
    public:
        GParts(GBndLib* bndlib=NULL);
        GParts(NPY<float>* partBuf, NPY<float>* tranBuf, NPY<float>* planBuf, const char* spec, GBndLib* bndlib=NULL);
        GParts(NPY<float>* partBuf, NPY<float>* tranBuf, NPY<float>* planBuf, GItemList* spec, GBndLib* bndlib=NULL);
   public:
        void setName(const char* name);
        void setBndLib(GBndLib* blib);
        void setVerbosity(unsigned verbosity); 
        void add(GParts* other, unsigned verbosity);
        void close();
        void enlargeBBox(unsigned int part, float epsilon=0.00001f);
        void enlargeBBoxAll(float epsilon=0.00001f);
    public:
        // transients for debugging convenience when made from NCSG
        void setCSG(const NCSG* csg);
        const NCSG* getCSG() const ;
    private:
        void init(const char* spec);        
        void init();        
    public: 
        const char*  getName();

        bool isPartList();
        bool isNodeTree();
        bool isInvisible();
        bool isClosed();
        bool isLoaded();

        void setInvisible();
        void setPartList();
        void setNodeTree();

        unsigned int getIndex(unsigned int part);
        unsigned int getTypeCode(unsigned int part);
        unsigned int getNodeIndex(unsigned int part);
        unsigned int getBoundary(unsigned int part);

        unsigned getAnalyticVersion();
        void setAnalyticVersion(unsigned vers);
    private: 
        void setLoaded(bool loaded=true);
    public: 
        std::string  getBoundaryName(unsigned int part);
        const char*  getTypeName(unsigned int part);
   private:
        nbbox        getBBox(unsigned int i);
        gfloat3      getGfloat3(unsigned int i, unsigned int j, unsigned int k);
        float*       getValues(unsigned int i, unsigned int j, unsigned int k);
    public:
        nivec4       getPrimInfo(unsigned int iprim);
   public:
        void setIndex(unsigned int part, unsigned int index);
        void setTypeCode(unsigned int part, unsigned int typecode);
        void setNodeIndex(unsigned int part, unsigned int nodeindex);
        void setBoundary(unsigned int part, unsigned int boundary);
   public:
        void setBoundaryAll(unsigned int boundary);
        void setNodeIndexAll(unsigned int nodeindex);
    public:
        GBndLib*           getBndLib();
        GItemList*         getBndSpec();
        unsigned int       getNumPrim();
        unsigned int       getNumParts();
        unsigned int       getPrimNumParts(unsigned int prim_index);
        std::string        desc(); 
    public:
        NPY<int>*          getPrimBuffer();
        NPY<float>*        getPartBuffer();
        NPY<float>*        getTranBuffer(); // inverse transforms IR*IT ie inverse of T*R 
        NPY<float>*        getPlanBuffer(); // planes used by convex polyhedra such as trapezoid
        NPY<float>*        getBuffer(const char* tag) const ;

    public:
        void fulldump(const char* msg="GParts::fulldump", unsigned lim=10 );
        void dump(const char* msg="GParts::dump", unsigned lim=10 );
        void dumpPrimInfo(const char* msg="GParts::dumpPrimInfo", unsigned lim=10 );
        void dumpPrimBuffer(const char* msg="GParts::dumpPrimBuffer");
        void Summary(const char* msg="GParts::Summary", unsigned lim=10 );
    private:
        void dumpPrim(unsigned primIdx);
    public:
        void setSensorSurface(const char* surface="lvPmtHemiCathodeSensorSurface");
        void setContainingMaterial(const char* material="MineralOil");
        void applyPlacementTransform(GMatrix<float>* placement, unsigned verbosity=0);

        void save(const char* dir);
        static GParts* Load(const char* dir);
    private:
        void registerBoundaries();  // convert the boundary spec names into integer codes using bndlib, setting into partBuffer
        void makePrimBuffer();
        void reconstructPartsPerPrim();
    private:
        void setBndSpec(GItemList* bndspec);
        void setPartBuffer(NPY<float>* part_buffer);
        void setPrimBuffer(NPY<int>*   prim_buffer);
        void setTranBuffer(NPY<float>* tran_buffer);
        void setPlanBuffer(NPY<float>* plan_buffer);
        void setPrimFlag(OpticksCSG_t primflag);
        OpticksCSG_t getPrimFlag(); 
        const char*  getPrimFlagString() const ; 

    private:
       unsigned int getUInt(unsigned int part, unsigned int j, unsigned int k);
       void         setUInt(unsigned int part, unsigned int j, unsigned int k, unsigned int value);
    private:
        // almost no state other than buffers, just icing on top of them
        // allowing this to copied/used on GPU in cu/hemi-pmt.cu
        NPY<float>*        m_part_buffer ; 
        NPY<float>*        m_tran_buffer ; 
        NPY<float>*        m_plan_buffer ; 
        GItemList*         m_bndspec ;  
        GBndLib*           m_bndlib ; 
        const char*        m_name ;         
    private:
        typedef std::map<unsigned, unsigned> MUU ; 
        typedef std::vector<unsigned> VU ; 
    private:
        NPY<int>*          m_prim_buffer ; 
        bool               m_closed ; 
        bool               m_loaded ; 
        MUU                m_parts_per_prim ;
        VU                 m_tran_per_add ; 
        VU                 m_part_per_add ; 
        VU                 m_plan_per_add ; 
        unsigned           m_verbosity ; 
        unsigned           m_analytic_version ; 
        OpticksCSG_t       m_primflag ; 
        const char*        m_medium ; 
        const NCSG*        m_csg ; 

};

#include "GGEO_TAIL.hh"


