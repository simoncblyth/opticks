#pragma once

#include <map>
#include <cassert>

struct guint4 ; 
struct gbbox ; 
struct gfloat3 ; 
class GItemList ; 
class GBndLib ; 

struct NSlice ; 
template <typename T> class NPY ;

// *GParts* 
//     creates *solid* buffer from the *parts* buffer
//     the *parts* buffer .npy is created by detdesc partitioning with pmt-/tree.py 
//

class GParts {
    public:
       // conventional names for interfacing
       static const char* CONTAINING_MATERIAL ; 
       static const char* SENSOR_SURFACE ; 
    public:
       // shapes with analytic intersection implementations 
       static const char* SPHERE_ ;
       static const char* TUBS_ ; 
       static const char* BOX_ ; 
       static const char* TypeName(unsigned int typecode);
       enum {
              ZERO, 
              SPHERE, 
              TUBS, 
              BOX
            };
    public:
        // buffer layout, must match locations in pmt-/tree.py:convert 
        enum { 
              QUADS_PER_ITEM = 4, 
              NJ = 4,
              NK = 4,
              SK = 4  
            } ;

        enum { INDEX_J  = 1, INDEX_K  = 1 };
        enum { BOUNDARY_J = 1, BOUNDARY_K = 2 };
        enum { FLAGS_J  = 1, FLAGS_K  = 3 };

        enum { BBMIN_J = 2,     BBMIN_K = 0 };
        enum { TYPECODE_J  = 2, TYPECODE_K  = 3 };

        enum { BBMAX_J = 3,     BBMAX_K = 0 };
        enum { NODEINDEX_J = 3, NODEINDEX_K = 3 };
    public:
        static GParts* makePart(gbbox& bb, const char* spec );
        static GParts* makeBox(gbbox& bb, const char* spec );
    public:
        GParts(GBndLib* bndlib=NULL);
        GParts(NPY<float>* buffer, const char* spec, GBndLib* bndlib=NULL);
        GParts(NPY<float>* buffer, GItemList* spec, GBndLib* bndlib=NULL);
    public:
        void setBndLib(GBndLib* blib);
        void add(GParts* other);
        void close();
        bool isClosed();
    private:
        void init(const char* spec);        
        void init();        
    public: 
        unsigned int getIndex(unsigned int part);
        unsigned int getFlags(unsigned int part);
        unsigned int getTypeCode(unsigned int part);
        unsigned int getNodeIndex(unsigned int part);
        unsigned int getBoundary(unsigned int part);
        std::string  getBoundaryName(unsigned int part);
        const char*  getTypeName(unsigned int part);
   private:
        gbbox        getBBox(unsigned int i);
        gfloat3      getGfloat3(unsigned int i, unsigned int j, unsigned int k);
    public:
        guint4       getSolidInfo(unsigned int isolid);
   public:
        void setTypeCode(unsigned int part, unsigned int typecode);
        void setNodeIndex(unsigned int part, unsigned int nodeindex);
        void setIndex(unsigned int part, unsigned int index);
        void setBoundary(unsigned int part, unsigned int boundary);
    public:
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
};



inline GParts::GParts(GBndLib* bndlib) 
      :
      m_part_buffer(NULL),
      m_bndspec(NULL),
      m_bndlib(bndlib),
      m_solid_buffer(NULL),
      m_closed(false)
{
      init() ; 
}

inline GParts::GParts(NPY<float>* buffer, const char* spec, GBndLib* bndlib) 
      :
      m_part_buffer(buffer),
      m_bndspec(NULL),
      m_bndlib(bndlib),
      m_solid_buffer(NULL),
      m_closed(false)
{
      init(spec) ; 
}
      
inline GParts::GParts(NPY<float>* buffer, GItemList* spec, GBndLib* bndlib) 
      :
      m_part_buffer(buffer),
      m_bndspec(spec),
      m_bndlib(bndlib),
      m_solid_buffer(NULL),
      m_closed(false)
{
      init() ; 
}



inline bool GParts::isClosed()
{
    return m_closed ; 
}



inline unsigned int GParts::getSolidNumParts(unsigned int solid_index)
{
    return m_parts_per_solid.count(solid_index)==1 ? m_parts_per_solid[solid_index] : 0 ; 
}


inline void GParts::setBndSpec(GItemList* bndspec)
{
    m_bndspec = bndspec ;
}
inline GItemList* GParts::getBndSpec()
{
    return m_bndspec ; 
}

inline void GParts::setBndLib(GBndLib* bndlib)
{
    m_bndlib = bndlib ; 
}


inline void GParts::setSolidBuffer(NPY<unsigned int>* solid_buffer)
{
    m_solid_buffer = solid_buffer ; 
}
inline NPY<unsigned int>* GParts::getSolidBuffer()
{
    return m_solid_buffer ; 
}

inline void GParts::setPartBuffer(NPY<float>* part_buffer)
{
    m_part_buffer = part_buffer ; 
}
inline NPY<float>* GParts::getPartBuffer()
{
    return m_part_buffer ; 
}




