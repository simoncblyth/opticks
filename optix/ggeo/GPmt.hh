#pragma once

#include <cstddef>

class GCache ; 
class GParts ; 
class GCSG ; 
class GBndLib ; 

struct gbbox ; 
struct NSlice ; 


class GPmt {
  public:
       static const char* FILENAME ;  
       static const char* FILENAME_CSG ;  
       static const char* GPMT ;  
   public:
       // loads persisted GParts buffer and associates with the GPmt
       static GPmt* load(GCache* cache, GBndLib* bndlib, unsigned int index, NSlice* slice=NULL);
   public:
       GPmt(GCache* cache, GBndLib* bndlib, unsigned int index);
   public:
       void addContainer(gbbox& bb, const char* bnd );
   private:
       void loadFromCache(NSlice* slice);    
       void setParts(GParts* parts);
       void setCSG(GCSG* csg);
   public:
       GParts* getParts();
       GCSG*   getCSG();
   private:
       GCache*            m_cache ; 
       GBndLib*           m_bndlib ; 
       unsigned int       m_index ;
       GParts*            m_parts ;
       GCSG*              m_csg ;
};


inline GPmt::GPmt(GCache* cache, GBndLib* bndlib, unsigned int index) 
    :
    m_cache(cache),
    m_bndlib(bndlib),
    m_index(index),
    m_parts(NULL),
    m_csg(NULL)
{
}

inline void GPmt::setParts(GParts* parts)
{
    m_parts = parts ; 
}
inline GParts* GPmt::getParts()
{
    return m_parts ; 
}

inline void GPmt::setCSG(GCSG* csg)
{
    m_csg = csg ; 
}
inline GCSG* GPmt::getCSG()
{
    return m_csg ; 
}


