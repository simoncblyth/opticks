#pragma once
#include <string>
#include "GPropertyMap.hh"

// Boundary Identity
// ~~~~~~~~~~~~~~~~~~~~
//
// Boundary indices are assigned by GBoundaryLib::get based on distinct property values
// this somewhat complicated approach to boundary identity is necessary 
// as GBoundary incorporates info from inner/outer material/surface so GBoundary 
// does not map to simple notions of identity it being a boundary between 
// materials with specific surfaces(or maybe no associated surface) 
//
// Where are boundary indice affixed to the triangles of the geometry ?
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//      GSolid::getBoundaryIndices 
//      GSolid::setBoundary 
//      GNode::setBoundaryIndices   repeats the indice for every triangle 
//
// Which surface is relevant (inner or outer) depends on light direction.  eg think about
// inside of SST tank as opposed to it outside. 
//
// OptiX has no notion of surface, so using OptiX materials to hold this info
//
//
class GBoundary {  
  public:
      static const char* imaterial ; 
      static const char* omaterial ; 
      static const char* isurface ; 
      static const char* osurface ; 
      static const char* iextra ; 
      static const char* oextra ; 

  public:
      GBoundary();
      GBoundary(
                 GPropertyMap<float>* imaterial, 
                 GPropertyMap<float>* omaterial, 
                 GPropertyMap<float>* isurface, 
                 GPropertyMap<float>* osurface,
                 GPropertyMap<float>* iextra,
                 GPropertyMap<float>* oextra
                 );
      virtual ~GBoundary();

  public:
      void Summary(const char* msg="GBoundary::Summary", unsigned int nline=0);
      char* pdigest(int ifr, int ito);
      std::string getPDigestString(int ifr, int ito);
      std::string description();

  public:
      unsigned int getIndex();
      void setIndex(unsigned int index);
      bool hasInnerMaterial(const char* shortname);
      bool hasOuterSensorSurface();
  public:
      void setInnerMaterial(GPropertyMap<float>* imaterial);
      void setOuterMaterial(GPropertyMap<float>* omaterial);
      void setInnerSurface(GPropertyMap<float>* isurface);
      void setOuterSurface(GPropertyMap<float>* osurface);
      void setInnerExtra(GPropertyMap<float>* iextra);
      void setOuterExtra(GPropertyMap<float>* oextra);

  public:
      GPropertyMap<float>* getInnerMaterial();
      GPropertyMap<float>* getOuterMaterial();
      GPropertyMap<float>* getInnerSurface();
      GPropertyMap<float>* getOuterSurface();
      GPropertyMap<float>* getInnerExtra();
      GPropertyMap<float>* getOuterExtra();

  public:
      GPropertyMap<float>* getConstituentByIndex(unsigned int p);
      static const char* getConstituentNameByIndex(unsigned int p);

  private:
      GPropertyMap<float>*  m_imaterial ; 
      GPropertyMap<float>*  m_omaterial ; 
      GPropertyMap<float>*  m_isurface ; 
      GPropertyMap<float>*  m_osurface ; 
      GPropertyMap<float>*  m_iextra ; 
      GPropertyMap<float>*  m_oextra ; 

  private:
      unsigned int m_index ; 


};


inline void GBoundary::setIndex(unsigned int index)
{
    m_index = index ;
}

inline unsigned int GBoundary::getIndex()
{
    return m_index ;
}



inline GPropertyMap<float>* GBoundary::getInnerMaterial()
{
    return m_imaterial ; 
}
inline GPropertyMap<float>* GBoundary::getOuterMaterial()
{
    return m_omaterial ; 
}
inline GPropertyMap<float>* GBoundary::getInnerSurface()
{
    return m_isurface ; 
}
inline GPropertyMap<float>* GBoundary::getOuterSurface()
{
    return m_osurface ; 
}
inline GPropertyMap<float>* GBoundary::getInnerExtra()
{
    return m_iextra ; 
}
inline GPropertyMap<float>* GBoundary::getOuterExtra()
{
    return m_oextra ; 
}






inline void GBoundary::setInnerMaterial(GPropertyMap<float>* imaterial)
{
    m_imaterial = imaterial ; 
}
inline void GBoundary::setOuterMaterial(GPropertyMap<float>* omaterial)
{
    m_omaterial = omaterial ; 
}
inline void GBoundary::setInnerSurface(GPropertyMap<float>* isurface)
{
    m_isurface = isurface ; 
}
inline void GBoundary::setOuterSurface(GPropertyMap<float>* osurface)
{
    m_osurface = osurface ; 
}
inline void GBoundary::setInnerExtra(GPropertyMap<float>* iextra)
{
    m_iextra = iextra ; 
}
inline void GBoundary::setOuterExtra(GPropertyMap<float>* oextra)
{
    m_oextra = oextra ; 
}



