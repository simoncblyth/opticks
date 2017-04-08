#pragma once

#include <map>
#include <string>
#include <vector>

#include <glm/fwd.hpp>
#include "OpticksCSG.h"

struct npart ; 
struct NSlice ; 
template <typename T> class NPY ;
class NCSG ; 

struct guint4 ; 
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
    public:
       // conventional names for interfacing
       static const char* CONTAINING_MATERIAL ; 
       static const char* SENSOR_SURFACE ; 
    public:
        // buffer layout, must match locations in pmt-/tree.py:convert 
        enum { 
              QUADS_PER_ITEM = 4, 
              NJ = 4,
              NK = 4,
              SK = 4  
            } ;
    public:
        static GParts* make(const npart& pt, const char* spec);
        static GParts* make(OpticksCSG_t csgflag, glm::vec4& param, const char* spec);
        static GParts* make(NCSG* tree);
    public:
        static GParts* combine(std::vector<GParts*> subs);
    public:
        GParts(GBndLib* bndlib=NULL);
        GParts(NPY<float>* partBuf, NPY<float>* iritBuf, const char* spec, GBndLib* bndlib=NULL);
        GParts(NPY<float>* partBuf, NPY<float>* iritBuf, GItemList* spec, GBndLib* bndlib=NULL);
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
        unsigned int getIndex(unsigned int part);
        unsigned int getTypeCode(unsigned int part);
        unsigned int getNodeIndex(unsigned int part);
        unsigned int getBoundary(unsigned int part);
    public: 
        /*
        unsigned int getFlags(unsigned int part);
        void setFlags(unsigned int part, unsigned int flags);
        void setFlagsAll(unsigned int flags);
        */
    public: 
        std::string  getBoundaryName(unsigned int part);
        const char*  getTypeName(unsigned int part);
   private:
        gbbox        getBBox(unsigned int i);
        gfloat3      getGfloat3(unsigned int i, unsigned int j, unsigned int k);
        float*       getValues(unsigned int i, unsigned int j, unsigned int k);
    public:
        guint4       getPrimInfo(unsigned int iprim);
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
        NPY<unsigned int>* getPrimBuffer();
        NPY<float>*        getPartBuffer();
        NPY<float>*        getIritBuffer(); // inverse transforms IR*IT ie inverse of T*R 
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
        void save(const char* dir);
    private:
        void registerBoundaries();
        void makePrimBuffer();
    private:
        void setBndSpec(GItemList* bndspec);
        void setPartBuffer(NPY<float>* part_buffer);
        void setPrimBuffer(NPY<unsigned int>* prim_buffer);
        void setIritBuffer(NPY<float>* irit_buffer);
    private:
       unsigned int getUInt(unsigned int part, unsigned int j, unsigned int k);
       void         setUInt(unsigned int part, unsigned int j, unsigned int k, unsigned int value);
    private:
        // almost no state other than buffers, just icing on top of them
        // allowing this to copied/used on GPU in cu/hemi-pmt.cu
        NPY<float>*        m_part_buffer ; 
        NPY<float>*        m_irit_buffer ; 
        GItemList*         m_bndspec ;  
        GBndLib*           m_bndlib ; 
        const char*        m_name ;         
    private:
        NPY<unsigned int>* m_prim_buffer ; 
        bool               m_closed ; 
        std::map<unsigned int, unsigned int> m_parts_per_prim ;
        std::map<unsigned int, unsigned int> m_flag_prim ;
        bool               m_verbose ; 
};

#include "GGEO_TAIL.hh"


