#pragma once

#include <vector>
#include "GPropertyLib.hh"


// skin and border surfaces have an associated optical surface 
// that is lodged inside GPropertyMap
// in addition to 1(for skin) or 2(for border) volume names
//     

class GOpticalSurface ; 
class GSkinSurface ; 
class GBorderSurface ; 
class GItemList ; 

class GSurfaceLib : public GPropertyLib {
   public:
       static const char* propertyName(unsigned int k);
       // 4 standard surface property names : interleaved into float4 wavelength texture
       static const char* detect ;
       static const char* absorb ;
       static const char* reflect_specular ;
       static const char* reflect_diffuse ;
  public:
       // some model-mismatch translation required for surface properties
       static const char* EFFICIENCY ; 
       static const char* REFLECTIVITY ; 
   public:
       static float        SURFACE_UNSET ; 
       static const char* keyspec ;
   public:
       static GSurfaceLib* load(GCache* cache);
       GSurfaceLib(GCache* cache); 
       void Summary(const char* msg="GSurfaceLib::Summary");
       void dump(const char* msg="GSurfaceLib::dump");
       void dump(GPropertyMap<float>* surf, const char* msg="GSurfaceLib::dump");
   private:
       void init();
   public:
       // concretization of GPropertyLib
       void defineDefaults(GPropertyMap<float>* defaults); 
  public:
      // methods for debug
      void setFakeEfficiency(float fake_efficiency);
   public:
       void add(GSkinSurface* ss);
       void add(GBorderSurface* bs);
       void add(GPropertyMap<float>* surf);
   public:
       GPropertyMap<float>* getSurface(unsigned int i);
   private:
       GPropertyMap<float>* createStandardSurface(GPropertyMap<float>* src);
       bool checkSurface( GPropertyMap<float>* surf);
   public:
       unsigned int getNumRawSurfaces();
       unsigned int getNumSurfaces();
   public:
       void createBuffer();
       void import();
   private:
       void import( GPropertyMap<float>* surf, float* data, unsigned int nj, unsigned int nk );
   private:
       std::vector<GPropertyMap<float>*>       m_surfaces_raw ; 
       std::vector<GPropertyMap<float>*>       m_surfaces ; 
       float                                   m_fake_efficiency ; 


};

inline GSurfaceLib::GSurfaceLib(GCache* cache) 
    :
    GPropertyLib(cache, "GSurfaceLib"),
    m_fake_efficiency(-1.f)
{
    init();
}
 
inline unsigned int GSurfaceLib::getNumSurfaces()
{
    return m_surfaces.size();
}
inline unsigned int GSurfaceLib::getNumRawSurfaces()
{
    return m_surfaces_raw.size();
}
inline GPropertyMap<float>* GSurfaceLib::getSurface(unsigned int i)
{
    return m_surfaces[i] ;
}
inline void GSurfaceLib::setFakeEfficiency(float fake_efficiency)
{
    m_fake_efficiency = fake_efficiency ; 
}

