#pragma once

#include <vector>
#include "GVector.hh"
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
  public:
       static const char* detect ;
       static const char* absorb ;
       static const char* reflect_specular ;
       static const char* reflect_diffuse ;
  public:
       static const char* extra_x ; 
       static const char* extra_y ; 
       static const char* extra_z ; 
       static const char* extra_w ; 
  public:
       // some model-mismatch translation required for surface properties
       static const char* EFFICIENCY ; 
       static const char* REFLECTIVITY ; 
   public:
       static const char*  SENSOR_SURFACE ;
       static float        SURFACE_UNSET ; 
       static const char* keyspec ;
   public:
       void save();
       static GSurfaceLib* load(GCache* cache);
   public:
       GSurfaceLib(GCache* cache); 
   public:
       void Summary(const char* msg="GSurfaceLib::Summary");
       void dump(const char* msg="GSurfaceLib::dump");
       void dump(GPropertyMap<float>* surf);
       void dump(GPropertyMap<float>* surf, const char* msg);
       void dump(unsigned int index);
   private:
       void init();
   public:
       // concretization of GPropertyLib
       void defineDefaults(GPropertyMap<float>* defaults); 
       NPY<float>* createBuffer();
       GItemList*  createNames();
   public:
      // methods for debug
       void setFakeEfficiency(float fake_efficiency);
       GPropertyMap<float>* makePerfect(const char* name, float detect_, float absorb_, float reflect_specular_, float reflect_diffuse_);
       void addPerfectSurfaces();
   public:
       void add(GSkinSurface* ss);
       void add(GBorderSurface* bs);
       void add(GPropertyMap<float>* surf);
   private:
       void addDirect(GPropertyMap<float>* surf);
   public:
       void sort();
       bool operator()(const GPropertyMap<float>* a_, const GPropertyMap<float>* b_);
   public:
       guint4               getOpticalSurface(unsigned int index);  // zero based index
       GPropertyMap<float>* getSurface(unsigned int index);         // zero based index
       GPropertyMap<float>* getSurface(const char* name);        
       GPropertyMap<float>* getSensorSurface(unsigned int offset=0);  // 0: first, 1:second 
       bool hasSurface(unsigned int index); 
       bool hasSurface(const char* name); 
   private:
       guint4               createOpticalSurface(GPropertyMap<float>* src);
       GPropertyMap<float>* createStandardSurface(GPropertyMap<float>* src);
       bool checkSurface( GPropertyMap<float>* surf);
   public:
      // unlike former GBoundaryLib optical buffer one this is surface only
       NPY<unsigned int>* createOpticalBuffer();  
       void importOpticalBuffer(NPY<unsigned int>* ibuf);
       void saveOpticalBuffer();
       void loadOpticalBuffer();
       void setOpticalBuffer(NPY<unsigned int>* ibuf);
       NPY<unsigned int>* getOpticalBuffer();
   public:
       unsigned int getNumSurfaces();
       bool isSensorSurface(unsigned int surface); // name suffix based, see AssimpGGeo::convertSensor
   public:
       void import();
   private:
       void import( GPropertyMap<float>* surf, float* data, unsigned int nj, unsigned int nk );
   private:
       std::vector<GPropertyMap<float>*>       m_surfaces ; 
       float                                   m_fake_efficiency ; 
       NPY<unsigned int>*                      m_optical_buffer ; 


};

inline GSurfaceLib::GSurfaceLib(GCache* cache) 
    :
    GPropertyLib(cache, "GSurfaceLib"),
    m_fake_efficiency(-1.f),
    m_optical_buffer(NULL)
{
    init();
}
 
inline unsigned int GSurfaceLib::getNumSurfaces()
{
    return m_surfaces.size();
}


inline void GSurfaceLib::setFakeEfficiency(float fake_efficiency)
{
    m_fake_efficiency = fake_efficiency ; 
}
inline void GSurfaceLib::setOpticalBuffer(NPY<unsigned int>* ibuf)
{
    m_optical_buffer = ibuf ; 
}
inline NPY<unsigned int>* GSurfaceLib::getOpticalBuffer()
{
    return m_optical_buffer ;
}

