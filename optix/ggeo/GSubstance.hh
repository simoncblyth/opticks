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
      void Summary(const char* msg="GSubstance::Summary", unsigned int nline=0);
      char* digest();

  public:
      // 
      // NB indices are assigned by GSubstanceLib::get based on distinct property values
      //    this somewhat complicated approach is necessary as GSubstance incorporates info 
      //    from inner/outer material/surface so GSubstance 
      //    does not map to simple notions of identity it being a boundary between 
      //    materials with specific surfaces(or maybe no associated surface) 
      //
      //    Where is the substance indice affixed to the triangles of the geometry ?
      //         GSolid::getSubstanceIndices GSolid::setSubstance GNode::setSubstanceIndices
      //    repeats the indice for every triangle 
      //
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

  public:
      void setTexProps(GPropertyMap* texprops);
      GPropertyMap* getTexProps();
      void dumpTexProps(const char* msg, double wavelength);

  private:
      GPropertyMap*  m_imaterial ; 
      GPropertyMap*  m_omaterial ; 
      GPropertyMap*  m_isurface ; 
      GPropertyMap*  m_osurface ; 

  private:
      GPropertyMap*  m_texprops ; 


      unsigned int m_index ; 


};

#endif
