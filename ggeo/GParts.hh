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
class NCSG ; 

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
    public:
        // buffer layout, must match locations in pmt-/tree.py:convert 
        enum { 
              QUADS_PER_ITEM = 4, 
              NJ = 4,
              NK = 4,
              SK = 4  
            } ;
    public:

       // hmm boundary spec is a node-type-qty, not a mesh-type-qty 
       // so it does not belong inside GParts (a mesh-type-qty)
       // ... there are relatively few mesh-type-qty for 
       // each distinct shape (~249), but much more node-type-qty (~12k)
       //
        static GParts* make(const npart& pt, const char* spec);
        static GParts* make(OpticksCSG_t csgflag, glm::vec4& param, const char* spec);
        static GParts* make(NCSG* tree, const char* spec);
    public:
        static GParts* combine(std::vector<GParts*> subs);
        static GParts* combine(GParts* onesub);   // for consistent handling between 1 and many 
    public:
        GParts(GBndLib* bndlib=NULL);
        GParts(NPY<float>* partBuf, NPY<float>* tranBuf, NPY<float>* planBuf, const char* spec, GBndLib* bndlib=NULL);
        GParts(NPY<float>* partBuf, NPY<float>* tranBuf, NPY<float>* planBuf, GItemList* spec, GBndLib* bndlib=NULL);
    public:
        void setName(const char* name);
        void setBndLib(GBndLib* blib);
        void setVerbose(bool verbose); 
        void add(GParts* other);
        void close();
        bool isClosed();
        void enlargeBBox(unsigned int part, float epsilon=0.00001f);
        void enlargeBBoxAll(float epsilon=0.00001f);
    private:
        void init(const char* spec);        
        void init();        
    public: 
        const char*  getName();
        bool isPartList();
        bool isNodeTree();

        unsigned int getIndex(unsigned int part);
        unsigned int getTypeCode(unsigned int part);
        unsigned int getNodeIndex(unsigned int part);
        unsigned int getBoundary(unsigned int part);

        unsigned getAnalyticVersion();
        void setAnalyticVersion(unsigned vers);
    public: 
        std::string  getBoundaryName(unsigned int part);
        const char*  getTypeName(unsigned int part);
   private:
        gbbox        getBBox(unsigned int i);
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
    public:
        NPY<int>*          getPrimBuffer();
        NPY<float>*        getPartBuffer();
        NPY<float>*        getTranBuffer(); // inverse transforms IR*IT ie inverse of T*R 
        NPY<float>*        getPlanBuffer(); // planes used by convex polyhedra such as trapezoid
    public:
        void fulldump(const char* msg="GParts::fulldump");
        void dump(const char* msg="GParts::dump");
        void dumpPrimInfo(const char* msg="GParts::dumpPrimInfo");
        void dumpPrimBuffer(const char* msg="GParts::dumpPrimBuffer");
        void Summary(const char* msg="GParts::Summary");
    private:
        void dumpPrim(unsigned primIdx);
    public:
        void setSensorSurface(const char* surface="lvPmtHemiCathodeSensorSurface");
        void setContainingMaterial(const char* material="MineralOil");
        void applyGlobalPlacementTransform(GMatrix<float>* placement, unsigned verbosity=0);
        void save(const char* dir);
    private:
        void registerBoundaries();
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
        MUU                m_parts_per_prim ;
        VU                 m_tran_per_add ; 
        VU                 m_part_per_add ; 
        VU                 m_plan_per_add ; 
        bool               m_verbose ; 
        unsigned           m_analytic_version ; 
        OpticksCSG_t       m_primflag ; 

};

#include "GGEO_TAIL.hh"


