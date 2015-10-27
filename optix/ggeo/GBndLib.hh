#pragma once

#include <vector>
#include <string>

#include "GVector.hh"
#include "GPropertyLib.hh"

template <typename T> class GPropertyMap ;
class GCache ; 
class GMaterialLib ; 
class GSurfaceLib ; 

//
// *GBndLib* differs from *GMaterialLib* and *GSurfaceLib* in that 
// creation of its float buffer needs to be deferred post cache 
// in order to allow dynamic addition of boundaries for eg analytic
// geometry inside-outs and for test boxes 
//
// Instead the index buffer is used for persisting  
// 

class GBndLib : public GPropertyLib {
  public:
       static unsigned int UNSET ; 
       static GBndLib* load(GCache* cache);
       GBndLib(GCache* cache);
  private:
       void init(); 
  public:
       unsigned int getNumBnd();
       std::string description(const guint4& bnd);
       std::string shortname(const guint4& bnd);
       bool contains(const guint4& bnd);
       unsigned int index(const guint4& bnd);
  public:
       guint4 parse( const char* spec, bool flip=false);
       bool contains( const char* spec, bool flip=false);
  public:
       // Bnd are only added if not already present
       void add(const guint4& bnd);
       guint4 add(const char* spec, bool flip=false);
       guint4 add(const char* imat, const char* omat, const char* isur, const char* osur);
       guint4 add(unsigned int imat, unsigned int omat, unsigned int isur, unsigned int osur);
  public:
       void importIndexBuffer();
       void saveIndexBuffer();
       void loadIndexBuffer();
       NPY<unsigned int>* createIndexBuffer();
       NPY<unsigned int>* getIndexBuffer();
       void setIndexBuffer(NPY<unsigned int>* ibuf);
  public:
       GItemList* createNames();
       NPY<float>* createBuffer();
       void import();
       void defineDefaults(GPropertyMap<float>* defaults);
  public:
       void dump(const char* msg="GBndLib::dump");
  public:
       void setMaterialLib(GMaterialLib* mlib);
       void setSurfaceLib(GSurfaceLib* slib);
  private:
       GMaterialLib*        m_mlib ; 
       GSurfaceLib*         m_slib ; 
       std::vector<guint4>  m_bnd ; 
       NPY<unsigned int>*   m_ibuf ;  
};


inline GBndLib::GBndLib(GCache* cache) 
   :
    GPropertyLib(cache, "GBndLib"),
    m_mlib(NULL),
    m_slib(NULL)
{
    init();
}

inline void GBndLib::setMaterialLib(GMaterialLib* mlib)
{
    m_mlib = mlib ;  
}
inline void GBndLib::setSurfaceLib(GSurfaceLib* slib)
{
    m_slib = slib ;  
}

inline unsigned int GBndLib::getNumBnd()
{
    return m_bnd.size() ; 
}

inline NPY<unsigned int>* GBndLib::getIndexBuffer()
{
    return m_ibuf ;
}
inline void GBndLib::setIndexBuffer(NPY<unsigned int>* ibuf)
{
    m_ibuf = ibuf ;
}


