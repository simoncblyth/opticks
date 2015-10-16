#pragma once

#include <map>
#include <cassert>
class GBuffer ; 

// source parts .npy created by pmt-/tree.py 

class GPmt {
   public:
       enum { QUADS_PER_ITEM = 4, 
              NJ = 4,
              NK = 4 } ;

       enum { TYPECODE_J = 2,   TYPECODE_K = 3 };
       enum { NODEINDEX_J = 3, NODEINDEX_K = 3 };

       static GPmt* load(const char* path="/tmp/pmt-hemi-parts.npy");
       GPmt(GBuffer* buf);
       void dump(const char* msg="GPmt::dump");
       void Summary(const char* msg="GPmt::Summary");
       unsigned int getNumSolids();
       unsigned int getSolidNumParts(unsigned int solid_index);
       unsigned int getNumParts();
       unsigned int getNodeIndex(unsigned int part_index);
       unsigned int getTypeCode(unsigned int part_index);
   private:
       void init();
       void setNumParts(unsigned int num_parts);
       void setNumSolids(unsigned int num_solids);
       unsigned int getUInt(unsigned int part_index, unsigned int j, unsigned int k);
   private:
       GBuffer* m_buffer ; 
       unsigned int m_num_solids ; 
       unsigned int m_num_parts ; 
       std::map<unsigned int, unsigned int> m_parts_per_solid ;

};

inline GPmt::GPmt(GBuffer* buffer) :
    m_buffer(buffer),
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



