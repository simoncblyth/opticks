#pragma once

#include <map>
#include <string>
#include <vector>

#include <glm/fwd.hpp>

struct npart ; 
struct NSlice ; 
template <typename T> class NPY ;

struct guint4 ; 
struct gbbox ; 
struct gfloat3 ; 

class GItemList ; 
class GBndLib ; 

// *GParts* 
//     creates *solid* buffer from the *parts* buffer
//     the *parts* buffer .npy is created by detdesc partitioning with pmt-/tree.py 
//

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GParts {
    public:
       // conventional names for interfacing
       static const char* CONTAINING_MATERIAL ; 
       static const char* SENSOR_SURFACE ; 
    public:
       // shapes with analytic intersection implementations 
       static const char* SPHERE_ ;
       static const char* TUBS_ ; 
       static const char* BOX_ ; 
       static const char* PRISM_ ; 
       static const char* TypeName(unsigned int typecode);

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
        static GParts* make(char typecode, glm::vec4& param, const char* spec);
    public:
        static GParts* combine(std::vector<GParts*> subs);
    public:
        GParts(GBndLib* bndlib=NULL);
        GParts(NPY<float>* buffer, const char* spec, GBndLib* bndlib=NULL);
        GParts(NPY<float>* buffer, GItemList* spec, GBndLib* bndlib=NULL);
    public:
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
        unsigned int getIndex(unsigned int part);
        unsigned int getFlags(unsigned int part);
        unsigned int getTypeCode(unsigned int part);
        unsigned int getNodeIndex(unsigned int part);
        unsigned int getBoundary(unsigned int part);
    public: 
        std::string  getBoundaryName(unsigned int part);
        const char*  getTypeName(unsigned int part);
   private:
        gbbox        getBBox(unsigned int i);
        gfloat3      getGfloat3(unsigned int i, unsigned int j, unsigned int k);
        float*       getValues(unsigned int i, unsigned int j, unsigned int k);
    public:
        guint4       getSolidInfo(unsigned int isolid);
   public:
        void setIndex(unsigned int part, unsigned int index);
        void setFlags(unsigned int part, unsigned int flags);
        void setTypeCode(unsigned int part, unsigned int typecode);
        void setNodeIndex(unsigned int part, unsigned int nodeindex);
        void setBoundary(unsigned int part, unsigned int boundary);
        void setBoundaryAll(unsigned int boundary);
    public:
        GBndLib*           getBndLib();
        GItemList*         getBndSpec();
        NPY<unsigned int>* getSolidBuffer();
        NPY<float>*        getPartBuffer();
        unsigned int       getNumSolids();
        unsigned int       getNumParts();
        unsigned int       getSolidNumParts(unsigned int solid_index);
    public:
        void dump(const char* msg="GPmt::dump");
        void dumpSolidInfo(const char* msg="GPmt::dumpSolidInfo");
        void Summary(const char* msg="GPmt::Summary");
    public:
        void setSensorSurface(const char* surface="lvPmtHemiCathodeSensorSurface");
        void setContainingMaterial(const char* material="MineralOil");
    private:
        void registerBoundaries();
        void makeSolidBuffer();
    private:
       void         setBndSpec(GItemList* bndspec);
       void         setPartBuffer(NPY<float>* part_buffer);
       void         setSolidBuffer(NPY<unsigned int>* solid_buffer);
       unsigned int getUInt(unsigned int part, unsigned int j, unsigned int k);
       void         setUInt(unsigned int part, unsigned int j, unsigned int k, unsigned int value);
    private:
        // almost no state other than buffers, just icing on top of them
        // allowing this to copied/used on GPU in cu/hemi-pmt.cu
        NPY<float>*        m_part_buffer ; 
        GItemList*         m_bndspec ;  
        GBndLib*           m_bndlib ; 
    private:
        NPY<unsigned int>* m_solid_buffer ; 
        bool               m_closed ; 
        std::map<unsigned int, unsigned int> m_parts_per_solid ;
        bool               m_verbose ; 
};

#include "GGEO_TAIL.hh"


