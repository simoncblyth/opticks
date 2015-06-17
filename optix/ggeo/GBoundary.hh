#pragma once
#include <string>
//
// Excluding "outer" material from GBoundary (more natural and prevents compounding)
// means that will always need to consider pairs of substances, 
// ordered according to light direction.
//
// Which surface is relevant (inner or outer) depends on light direction.  eg think about
// inside of SST tank as opposed to it outside. 
//
//  
//  Questions:
//
//  * how many distinct substances ?
//  * what is substance identity ?
//     
//    * may via a hash of all the properties of the PropertyMaps
// 
//   OptiX has no notion of surface, so using GBoundary to hold 
//   surface and material properties 
//


#include "GPropertyMap.hh"

class GOpticalSurface ; 



// hmm GBoundary, a better name would be GBoundary now ?
// 
// Boundary indices are assigned by GBoundaryLib::get based on distinct property values
// this somewhat complicated approach is necessary as GBoundary incorporates info 
// from inner/outer material/surface so GBoundary 
// does not map to simple notions of identity it being a boundary between 
// materials with specific surfaces(or maybe no associated surface) 
//
// Where is the voundary indice affixed to the triangles of the geometry ?
//      GSolid::getBoundaryIndices GSolid::setBoundary GNode::setBoundaryIndices
// repeats the indice for every triangle 
//

class GBoundary {   // TODO: rename together with associated classes to GBoundary et al
  public:
      static const char* imaterial ; 
      static const char* omaterial ; 
      static const char* isurface ; 
      static const char* osurface ; 
      static const char* iextra ; 
      static const char* oextra ; 
      static const char* inner_optical ; 
      static const char* outer_optical ; 

  public:
      GBoundary();
      GBoundary(
                 GPropertyMap<float>* imaterial, 
                 GPropertyMap<float>* omaterial, 
                 GPropertyMap<float>* isurface, 
                 GPropertyMap<float>* osurface,
                 GPropertyMap<float>* iextra,
                 GPropertyMap<float>* oextra,
                 GOpticalSurface*     inner_optical,
                 GOpticalSurface*     outer_optical
                 );
      virtual ~GBoundary();

  public:
      void Summary(const char* msg="GBoundary::Summary", unsigned int nline=0);
      //char* digest();
      char* pdigest(int ifr, int ito);
      std::string getPDigestString(int ifr, int ito);

  public:
      unsigned int getIndex();
      void setIndex(unsigned int index);

  public:
      void setInnerMaterial(GPropertyMap<float>* imaterial);
      void setOuterMaterial(GPropertyMap<float>* omaterial);
      void setInnerSurface(GPropertyMap<float>* isurface);
      void setOuterSurface(GPropertyMap<float>* osurface);
      void setInnerExtra(GPropertyMap<float>* iextra);
      void setOuterExtra(GPropertyMap<float>* oextra);
      void setInnerOptical(GOpticalSurface* ioptical);
      void setOuterOptical(GOpticalSurface* ooptical);



  public:
      GPropertyMap<float>* getInnerMaterial();
      GPropertyMap<float>* getOuterMaterial();
      GPropertyMap<float>* getInnerSurface();
      GPropertyMap<float>* getOuterSurface();
      GPropertyMap<float>* getInnerExtra();
      GPropertyMap<float>* getOuterExtra();

      GOpticalSurface*     getInnerOptical();
      GOpticalSurface*     getOuterOptical();



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
      GOpticalSurface*      m_inner_optical ; 
      GOpticalSurface*      m_outer_optical ; 


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

inline GOpticalSurface* GBoundary::getInnerOptical()
{
   return m_inner_optical ; 
}
inline GOpticalSurface* GBoundary::getOuterOptical()
{
   return m_outer_optical ; 
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


inline void GBoundary::setInnerOptical(GOpticalSurface* inner_optical)
{
    m_inner_optical = inner_optical ; 
}
inline void GBoundary::setOuterOptical(GOpticalSurface* outer_optical)
{
    m_outer_optical = outer_optical ; 
}



