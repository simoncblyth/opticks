#ifndef GSUBSTANCE_H
#define GSUBSTANCE_H

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

class GPropertyMap ;

// hmm GSubstance, a better name would be GBoundary now ?
class GSubstance {
  public:
      GSubstance(GPropertyMap* imaterial, GPropertyMap* omaterial, GPropertyMap* isurface, GPropertyMap* osurface );
      virtual ~GSubstance();

  public:
      void Summary(const char* msg="GSubstance::Summary");
      char* digest();

  public:
      unsigned int getIndex();
      void setIndex(unsigned int index);

  public:
      void setInnerMaterial(GPropertyMap* imaterial);
      void setOuterMaterial(GPropertyMap* omaterial);
      void setInnerSurface(GPropertyMap* isurface);
      void setOuterSurface(GPropertyMap* osurface);

      GPropertyMap* getInnerMaterial();
      GPropertyMap* getOuterMaterial();
      GPropertyMap* getInnerSurface();
      GPropertyMap* getOuterSurface();

  private:
      GPropertyMap*  m_imaterial ; 
      GPropertyMap*  m_omaterial ; 
      GPropertyMap*  m_isurface ; 
      GPropertyMap*  m_osurface ; 

      unsigned int m_index ; 


};

#endif
