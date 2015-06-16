#pragma once
#include <string>
//
// Excluding "outer" material from GSubstance (more natural and prevents compounding)
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
//   OptiX has no notion of surface, so using GSubstance to hold 
//   surface and material properties 
//


#include "GPropertyMap.hh"

class GOpticalSurface ; 



// hmm GSubstance, a better name would be GBoundary now ?
// 
// Substance indices are assigned by GSubstanceLib::get based on distinct property values
// this somewhat complicated approach is necessary as GSubstance incorporates info 
// from inner/outer material/surface so GSubstance 
// does not map to simple notions of identity it being a boundary between 
// materials with specific surfaces(or maybe no associated surface) 
//
// Where is the substance indice affixed to the triangles of the geometry ?
//      GSolid::getSubstanceIndices GSolid::setSubstance GNode::setSubstanceIndices
// repeats the indice for every triangle 
//

class GSubstance {   // TODO: rename together with associated classes to GBoundary et al
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
      GSubstance();
      GSubstance(
                 GPropertyMap<float>* imaterial, 
                 GPropertyMap<float>* omaterial, 
                 GPropertyMap<float>* isurface, 
                 GPropertyMap<float>* osurface,
                 GPropertyMap<float>* iextra,
                 GPropertyMap<float>* oextra,
                 GOpticalSurface*     inner_optical,
                 GOpticalSurface*     outer_optical
                 );
      virtual ~GSubstance();

  public:
      void Summary(const char* msg="GSubstance::Summary", unsigned int nline=0);
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

  public:
      //void setInnerOptical(GOpticalSurface* ioptical);
      //void setOuterOptical(GOpticalSurface* ooptical);



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


inline void GSubstance::setIndex(unsigned int index)
{
    m_index = index ;
}

inline unsigned int GSubstance::getIndex()
{
    return m_index ;
}




inline GOpticalSurface* GSubstance::getInnerOptical()
{
   return m_inner_optical ; 
}
inline GOpticalSurface* GSubstance::getOuterOptical()
{
   return m_outer_optical ; 
}



