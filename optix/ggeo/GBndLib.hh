#pragma once

#include <vector>
#include <string>

#include "GVector.hh"
#include "GPropertyLib.hh"

template <typename T> class GPropertyMap ;
class GCache ; 
class GMaterialLib ; 
class GSurfaceLib ; 
//class GBnd ; 

//
// *GBndLib* differs from *GMaterialLib* and *GSurfaceLib* in that 
// creation of its float buffer needs to be deferred post cache 
// in order to allow dynamic addition of boundaries for eg analytic
// geometry inside-outs and for test boxes 
//
// Instead the index buffer is used for persisting, which contains
// indices of materials and surfaces imat/omat/isur/osur 
//
// The boundary buffer is created dynamically by pulling the 
// relevant bytes from the material and surface libs. 
// 
// Former *GBoundaryLib* encompassed uint4 optical_buffer that 
// contained surface properties from GOpticalSurface, that
// is now moved to *GSurfaceLib*   
//

class GBndLib : public GPropertyLib {
  public:
       void save();
       static GBndLib* load(GCache* cache);
  public:
       GBndLib(GCache* cache);
  private:
       void init(); 
  public:
       unsigned int getNumBnd();
  public:
       std::string description(const guint4& bnd);
       std::string shortname(const guint4& bnd);
       bool contains(const guint4& bnd);
       unsigned int index(const guint4& bnd);
  public:
       // boundary index lookups
       guint4 getBnd(unsigned int boundary);
       unsigned int getInnerMaterial(unsigned int boundary);
       unsigned int getOuterMaterial(unsigned int boundary);
       unsigned int getInnerSurface(unsigned int boundary);
       unsigned int getOuterSurface(unsigned int boundary);
  public:
       guint4 parse( const char* spec, bool flip=false);
       bool contains( const char* spec, bool flip=false);
  public:
       unsigned int addBoundary( const char* imat, const char* omat, const char* isur, const char* osur) ;
  private:
       // Bnd are only added if not already present
       void add(const guint4& bnd);
       guint4 add(const char* spec, bool flip=false);
       guint4 add(const char* imat, const char* omat, const char* isur, const char* osur);
       guint4 add(unsigned int imat, unsigned int omat, unsigned int isur, unsigned int osur);
  public:
       void loadIndexBuffer();
       void importIndexBuffer();
  public:
       void saveIndexBuffer();
       void saveOpticalBuffer();
  public:
       NPY<unsigned int>* createIndexBuffer();
       NPY<unsigned int>* createOpticalBuffer();
  public:
       NPY<unsigned int>* getIndexBuffer();
       NPY<unsigned int>* getOpticalBuffer();
  public:
       void setIndexBuffer(NPY<unsigned int>* index_buffer);
       void setOpticalBuffer(NPY<unsigned int>* optical_buffer);
  public:
       unsigned int getMaterialLine(const char* shortname);
       static unsigned int getLine(unsigned int ibnd, unsigned int iquad);
       unsigned int getLineMin();
       unsigned int getLineMax();
  public:
       void createDynamicBuffers();
  public:
       GItemList* createNames();
       NPY<float>* createBuffer();
       void import();
       void sort();
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
       NPY<unsigned int>*   m_index_buffer ;  
       NPY<unsigned int>*   m_optical_buffer ;  
};


inline GBndLib::GBndLib(GCache* cache) 
   :
    GPropertyLib(cache, "GBndLib"),
    m_mlib(NULL),
    m_slib(NULL),
    m_index_buffer(NULL),
    m_optical_buffer(NULL)
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
    return m_index_buffer ;
}
inline void GBndLib::setIndexBuffer(NPY<unsigned int>* index_buffer)
{
    m_index_buffer = index_buffer ;
}

inline NPY<unsigned int>* GBndLib::getOpticalBuffer()
{
    return m_optical_buffer ;
}
inline void GBndLib::setOpticalBuffer(NPY<unsigned int>* optical_buffer)
{
    m_optical_buffer = optical_buffer ;
}


