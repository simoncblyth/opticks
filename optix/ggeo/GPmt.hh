#pragma once

#include <map>
#include <cassert>

class GCache ; 
class GBuffer ; 

// *GPmt::init* 
//
//      creates *solid* buffer from the *parts* buffer
//      the *parts* buffer .npy is created by detdesc partitioning with pmt-/tree.py 
//

class GPmt {
   public:
       enum { QUADS_PER_ITEM = 4, 
              NJ = 4,
              NK = 4 } ;

       // below must match pmt-/tree.py:copy_parts 
       enum { INDEX_J  = 1, INDEX_K  = 1 };
       enum { PARENT_J = 1, PARENT_K = 2 };
       enum { FLAGS_J  = 1, FLAGS_K  = 3 };
       enum { TYPECODE_J  = 2, TYPECODE_K  = 3 };
       enum { NODEINDEX_J = 3, NODEINDEX_K = 3 };

       static const char* FILENAME ;  
       static const char* GPMT ;  
       static const char* SPHERE_ ;
       static const char* TUBS_ ; 
       static const char* TypeName(unsigned int typecode);
   public:
       static GPmt* load(GCache* cache, unsigned int index=0);
       static GPmt* load(const char* path="/tmp/pmt-hemi-parts.npy", unsigned int index=0);
       GPmt(GCache* cache, unsigned int index=0);
       GPmt(GBuffer* part_buffer, unsigned int index=0);
   public:
       GBuffer* getSolidBuffer();
       GBuffer* getPartBuffer();
       void dump(const char* msg="GPmt::dump");
       void Summary(const char* msg="GPmt::Summary");
       unsigned int getNumSolids();
       unsigned int getSolidNumParts(unsigned int solid_index);
       unsigned int getNumParts();
   public: 
       unsigned int getIndex(unsigned int part_index);
       unsigned int getParent(unsigned int part_index);
       unsigned int getFlags(unsigned int part_index);
       unsigned int getTypeCode(unsigned int part_index);
       unsigned int getNodeIndex(unsigned int part_index);
   public:
       const char*  getTypeName(unsigned int part_index);
   private:
       void init();
       void setNumParts(unsigned int num_parts);
       void setNumSolids(unsigned int num_solids);
       void setSolidBuffer(GBuffer* solid_buffer);
       unsigned int getUInt(unsigned int part_index, unsigned int j, unsigned int k);
   private:
       GCache*      m_cache ; 
       unsigned int m_index ;
       GBuffer*     m_part_buffer ; 
       GBuffer*     m_solid_buffer ; 
       unsigned int m_num_solids ; 
       unsigned int m_num_parts ; 
       std::map<unsigned int, unsigned int> m_parts_per_solid ;

};


inline GPmt::GPmt(GCache* cache, unsigned int index) 
    :
    m_cache(cache),
    m_index(index),
    m_part_buffer(NULL),
    m_solid_buffer(NULL),
    m_num_solids(0),
    m_num_parts(0)
{
   init();
}

inline GPmt::GPmt(GBuffer* part_buffer, unsigned int index) 
    :
    m_cache(NULL),
    m_index(index),
    m_part_buffer(part_buffer),
    m_solid_buffer(NULL),
    m_num_solids(0),
    m_num_parts(0)
{
   init();
}

inline unsigned int GPmt::getNumSolids()
{
    return m_num_solids ; 
}
inline unsigned int GPmt::getNumParts()
{
    return m_num_parts ; 
}
inline void GPmt::setNumParts(unsigned int num_parts)
{
    m_num_parts = num_parts ; 
}
inline void GPmt::setNumSolids(unsigned int num_solids)
{
    m_num_solids = num_solids ; 
}

inline unsigned int GPmt::getSolidNumParts(unsigned int solid_index)
{
    return m_parts_per_solid.count(solid_index)==1 ? m_parts_per_solid[solid_index] : 0 ; 
}


inline void GPmt::setSolidBuffer(GBuffer* solid_buffer)
{
    m_solid_buffer = solid_buffer ; 
}
inline GBuffer* GPmt::getSolidBuffer()
{
    return m_solid_buffer ; 
}
inline GBuffer* GPmt::getPartBuffer()
{
    return m_part_buffer ; 
}

