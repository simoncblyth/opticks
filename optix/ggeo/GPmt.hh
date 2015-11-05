#pragma once

#include <map>
#include <cassert>

class GCache ; 

struct gbbox ; 
struct gfloat3 ; 
struct NSlice ; 

template <typename T> class NPY ;

// *GPmt::init* 
//
//      creates *solid* buffer from the *parts* buffer
//      the *parts* buffer .npy is created by detdesc partitioning with pmt-/tree.py 
//

class GPmt {
   public:
       enum {
              ZERO, 
              SPHERE, 
              TUBS, 
              BOX
            };

       static const char* SPHERE_ ;
       static const char* TUBS_ ; 
       static const char* BOX_ ; 

       static const char* TypeName(unsigned int typecode);
   public:
       enum { QUADS_PER_ITEM = 4, 
              NJ = 4,
              NK = 4 } ;

       // below must match pmt-/tree.py:copy_parts 
       enum { INDEX_J  = 1, INDEX_K  = 1 };
       enum { PARENT_J = 1, PARENT_K = 2 };
       enum { FLAGS_J  = 1, FLAGS_K  = 3 };
       enum { BBMIN_J = 2, BBMIN_K = 0 };
       enum { TYPECODE_J  = 2, TYPECODE_K  = 3 };
       enum { BBMAX_J = 3, BBMAX_K = 0 };
       enum { NODEINDEX_J = 3, NODEINDEX_K = 3 };

       static const char* FILENAME ;  
       static const char* GPMT ;  
   public:
       static GPmt* copy_with_container(GPmt* src, float container_factor=3.0f);
       static GPmt* load(GCache* cache, unsigned int index=0, NSlice* slice=NULL);
       GPmt(GCache* cache, unsigned int index=0);
       void make_container();
   private:
       void loadFromCache(NSlice* slice);   
   public:
       NPY<unsigned int>* getSolidBuffer();
       NPY<float>*        getPartBuffer();
       unsigned int       getNumSolids();
       unsigned int       getNumParts();
       unsigned int       getSolidNumParts(unsigned int solid_index);
   public:
       void dump(const char* msg="GPmt::dump");
       void Summary(const char* msg="GPmt::Summary");
   public: 
       unsigned int getIndex(unsigned int part_index);
       unsigned int getParent(unsigned int part_index);
       unsigned int getFlags(unsigned int part_index);
       unsigned int getTypeCode(unsigned int part_index);
       unsigned int getNodeIndex(unsigned int part_index);
       gbbox        getBBox(unsigned int i);
       gfloat3      getGfloat3(unsigned int i, unsigned int j, unsigned int k);
   public:
       const char*  getTypeName(unsigned int part_index);
   private:
       void         import();
       void         setPartBuffer(NPY<float>* part_buffer);
       void         setSolidBuffer(NPY<unsigned int>* solid_buffer);
       unsigned int getUInt(unsigned int part_index, unsigned int j, unsigned int k);
   private:
       GCache*            m_cache ; 
       unsigned int       m_index ;
       NPY<float>*        m_part_buffer ; 
       NPY<unsigned int>* m_solid_buffer ; 
       std::map<unsigned int, unsigned int> m_parts_per_solid ;

};


inline GPmt::GPmt(GCache* cache, unsigned int index) 
    :
    m_cache(cache),
    m_index(index),
    m_part_buffer(NULL),
    m_solid_buffer(NULL)
{
}

inline unsigned int GPmt::getSolidNumParts(unsigned int solid_index)
{
    return m_parts_per_solid.count(solid_index)==1 ? m_parts_per_solid[solid_index] : 0 ; 
}

inline void GPmt::setSolidBuffer(NPY<unsigned int>* solid_buffer)
{
    m_solid_buffer = solid_buffer ; 
}
inline NPY<unsigned int>* GPmt::getSolidBuffer()
{
    return m_solid_buffer ; 
}

inline void GPmt::setPartBuffer(NPY<float>* part_buffer)
{
    m_part_buffer = part_buffer ; 
}
inline NPY<float>* GPmt::getPartBuffer()
{
    return m_part_buffer ; 
}

