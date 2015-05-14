#ifndef GSUBSTANCELIB_H
#define GSUBSTANCELIB_H

#include <map>
#include <vector>
#include <string>


#include "GDomain.hh"
#include "GProperty.hh"
#include "GPropertyMap.hh"

class GSubstance ; 
class GBuffer ; 
class GSubstanceLibMetadata ; 
class GBuffer ; 

class GSubstanceLib {
  public:
    // standard property prefixes
    static const char* inner; 
    static const char* outer; 
  public:
    static unsigned int NUM_QUAD ; 
    static unsigned int DOMAIN_LENGTH ; 
    static float        DOMAIN_LOW ; 
    static float        DOMAIN_HIGH ; 
    static float        DOMAIN_STEP ; 
  public:
    // 4 standard material property names : interleaved into float4 wavelength texture
    static const char* refractive_index ; 
    static const char* absorption_length ; 
    static const char* scattering_length ; 
    static const char* reemission_prob ; 
  public:
    // 4 standard surface property names : interleaved into float4 wavelength texture
    static const char* detect ;
    static const char* absorb ;
    static const char* reflect_specular ;
    static const char* reflect_diffuse ;
  public:
     // extra
    static const char* reemission_cdf ; 
    static const char* extra_y ; 
    static const char* extra_z ; 
    static const char* extra_w ; 

  public:
     // workings for extra
    static const char* slow_component; 
    static const char* fast_component; 

  public:
    static const char* keymap ;

  private:
    // remission handling : needs energywise 1/wavelength[::-1] domain for CDF 
    static const char* scintillators ;
    static std::vector<std::string>* vscintillators ;
    static const char* reemissionkey ;
    static std::vector<std::string>* vreemissionkey;

    static GProperty<float>* reemission_prop ; 
    static std::string*      reemission_prop_digest ;  
  public:
      GSubstanceLib();
      virtual ~GSubstanceLib();

  public:
      // primary methods : lifecycle
      void setStandardDomain(GDomain<float>* standard_domain);
      GSubstance* get(
                      GPropertyMap<float>* imaterial, 
                      GPropertyMap<float>* omaterial, 
                      GPropertyMap<float>* isurface, 
                      GPropertyMap<float>* osurface,
                      GPropertyMap<float>* iextra, 
                      GPropertyMap<float>* oextra 
                 );
      GBuffer*      createWavelengthBuffer();  
      static unsigned int getLine(unsigned int isub, unsigned int ioff);

  public:
      // reemission handling 
      bool isScintillator(std::string& matShortName);
      bool isReemissionKey(std::string& lkey);
      void collectReemissionProp(GPropertyMap<float>* pmap);
      GProperty<float>* constructReemissionCDF(GPropertyMap<float>* pmap);
  public:
      // primary methods : querying 
      const char* getLocalKey(const char* dkey); // map standard -> local keys 
      unsigned int getNumSubstances();
      GSubstance* getSubstance(unsigned int index); 
      GSubstanceLibMetadata* getMetadata(); // populated by createWavelengthBuffer
      void Summary(const char* msg="GSubstanceLib::Summary");
      const char* getDigest(unsigned int index);
      

  public:
      // convenience methods
      void setWavelengthBuffer(GBuffer* buffer);
      GBuffer* getWavelengthBuffer();
      std::vector<std::string> splitString(std::string keys);

  private:
      // used for by "get" for standardization of substances, ready for serializing into wavelengthBuffer
      GSubstance* createStandardSubstance(GSubstance* substance);
      void standardizeMaterialProperties(GPropertyMap<float>* pstd, GPropertyMap<float>* pmap, const char* prefix);
      void standardizeSurfaceProperties(GPropertyMap<float>* pstd, GPropertyMap<float>* pmap, const char* prefix);
      void standardizeExtraProperties(GPropertyMap<float>* pstd, GPropertyMap<float>* pmap, const char* prefix);

      GProperty<float>* getPropertyOrDefault(GPropertyMap<float>* pmap, const char* pname);
      GProperty<float>* getProperty(GPropertyMap<float>* pmap, const char* dkey);

  private:
      // support for standardization 
      void defineDefaults(GPropertyMap<float>* defaults);
      void setDefaults(GPropertyMap<float>* defaults);
      GPropertyMap<float>* getDefaults();
      GProperty<float>* getDefaultProperty(const char* name);
      GProperty<float>* getRamp();
      void setKeyMap(const char* spec);

      char*  digest(std::vector<GProperty<float>*>& props);
      std::string digestString(std::vector<GProperty<float>*>& props);

      void digestDebug(GSubstance* substance, unsigned int isub);

  public:
      // another classes need access to "shape" of the standardization
      GDomain<float>*        getStandardDomain();
      static GDomain<float>* getDefaultDomain();
      unsigned int            getStandardDomainLength();

  public:
      void setMetadata(GSubstanceLibMetadata* meta); 

  public:
      void          dumpWavelengthBuffer(int wline=-1);
      static void   dumpWavelengthBuffer(int wline, GBuffer* buffer, GSubstanceLibMetadata* metadata, unsigned int numSubstance, unsigned int domainLength);

      GPropertyMap<float>* createStandardProperties(const char* name, GSubstance* substance);
      void checkMaterialProperties(GPropertyMap<float>* ptex, unsigned int offset, const char* prefix);
      void checkSurfaceProperties(GPropertyMap<float>* ptex, unsigned int offset, const char* prefix);
      void checkExtraProperties(GPropertyMap<float>* ptex, unsigned int offset, const char* prefix);

  public:
      static unsigned int getNumProp();
      static unsigned int getNumQuad();
      const char* materialPropertyName(unsigned int i);
      const char* surfacePropertyName(unsigned int i);
      const char* extraPropertyName(unsigned int i);
      char* propertyName(unsigned int p, unsigned int i);
      std::string propertyNameString(unsigned int p, unsigned int i);

  public:
      static GSubstanceLib* load(const char* dir);
      void loadWavelengthBuffer(GBuffer* buffer);
      GSubstance* loadSubstance(float* subData, unsigned int isub);


  private:
      std::map<std::string, std::string>   m_keymap ; //  
      std::map<std::string, GSubstance*>   m_registry ; 
      std::vector<std::string>             m_keys ; 

      bool                   m_standard ; // transitional : keeping this set to true
      unsigned int           m_num_quad ; 
      GDomain<float>*        m_standard_domain ;  
      GPropertyMap<float>*   m_defaults ;  
      GProperty<float>*      m_ramp ;  
      GSubstanceLibMetadata* m_meta ;
      GBuffer*               m_wavelength_buffer ;

};


inline GSubstanceLibMetadata* GSubstanceLib::getMetadata()
{
    return m_meta ; 
}
inline void GSubstanceLib::setMetadata(GSubstanceLibMetadata* meta)
{
    m_meta = meta ; 
}

inline GBuffer* GSubstanceLib::getWavelengthBuffer()
{
    return m_wavelength_buffer ; 
}
inline void GSubstanceLib::setWavelengthBuffer(GBuffer* wavelength_buffer)
{
    m_wavelength_buffer = wavelength_buffer ; 
}


inline unsigned int GSubstanceLib::getNumProp()
{
    return NUM_QUAD*4 ; 
}
inline unsigned int GSubstanceLib::getNumQuad()
{
    return NUM_QUAD ; 
}


inline void GSubstanceLib::setDefaults(GPropertyMap<float>* defaults)
{
    m_defaults = defaults ;
}
inline GPropertyMap<float>* GSubstanceLib::getDefaults()
{
    return m_defaults ;
}
inline GProperty<float>* GSubstanceLib::getRamp()
{
   return m_ramp ;
}
inline GDomain<float>* GSubstanceLib::getStandardDomain()
{
    return m_standard_domain ;
}



#endif
