#pragma once

//
// aiming to replace GBoundaryLib with a simpler and more flexible
// approach based on GMaterialLib and GSurfaceLib
// with post-cache deferred creation of boundary and optical buffers 
// allowing dynamic boundary creation from material, surface names
//

#include <set>

template <typename T> class GPropertyMap ;

#include "GVector.hh"

class GCache ; 
class GMaterialLib ; 
class GSurfaceLib ; 

class GBndLib {
  public:
       GBndLib(GCache* cache);
  public:
       void setMaterialLib(GMaterialLib* mlib);
       void setSurfaceLib(GSurfaceLib* slib);
       guint4 getOrCreate(
               GPropertyMap<float>* imat,  
               GPropertyMap<float>* omat,  
               GPropertyMap<float>* isur,  
               GPropertyMap<float>* osur);  
  private:
       GCache*              m_cache ; 
       GMaterialLib*        m_mlib ; 
       GSurfaceLib*         m_slib ; 
       std::set<guint4>     m_bnd ; 
};


inline GBndLib::GBndLib(GCache* cache) 
   :
    m_cache(cache),
    m_mlib(NULL),
    m_slib(NULL)
{
}

inline void GBndLib::setMaterialLib(GMaterialLib* mlib)
{
    m_mlib = mlib ;  
}
inline void GBndLib::setSurfaceLib(GSurfaceLib* slib)
{
    m_slib = slib ;  
}


