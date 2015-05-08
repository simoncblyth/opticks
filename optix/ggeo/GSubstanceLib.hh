#ifndef GSUBSTANCELIB_H
#define GSUBSTANCELIB_H

#include <map>
#include <vector>
#include <string>


#include "GDomain.hh"
#include "GProperty.hh"

class GSubstance ; 
class GPropertyMap ; 
class GBuffer ; 
class GSubstanceLibMetadata ; 

class GSubstanceLib {
  public:
    // standard property prefixes
    static const char* inner; 
    static const char* outer; 
  public:
    static unsigned int DOMAIN_LENGTH ; 
    static float        DOMAIN_LOW ; 
    static float        DOMAIN_HIGH ; 
    static float        DOMAIN_STEP ; 
  public:
    // standard material property names
    static const char* refractive_index ; 
    static const char* absorption_length ; 
    static const char* scattering_length ; 
    static const char* reemission_prob ; 
  public:
    // standard surface property names
    static const char* detect ;
    static const char* absorb ;
    static const char* reflect_specular ;
    static const char* reflect_diffuse ;
  public:
    static const char* keymap ;

  public:
      GSubstanceLib();
      virtual ~GSubstanceLib();

  public:
      // primary methods : lifecycle
      void setStandardDomain(GDomain<double>* standard_domain);
      GSubstance* get(GPropertyMap* imaterial, GPropertyMap* omaterial, GPropertyMap* isurface, GPropertyMap* osurface );
      GBuffer*      createWavelengthBuffer();  

  public:
      // primary methods : querying 
      const char* getLocalKey(const char* dkey); // map standard -> local keys 
      unsigned int getNumSubstances();
      GSubstance* getSubstance(unsigned int index); 
      GSubstanceLibMetadata* getMetadata(); // populated by createWavelengthBuffer
      void Summary(const char* msg="GSubstanceLib::Summary");

  private:
      // used for by "get" for standardization of substances, ready for serializing into wavelengthBuffer
      GSubstance* createStandardSubstance(GSubstance* substance);
      void addMaterialProperties(GPropertyMap* pstd, GPropertyMap* pmap, const char* prefix);
      void addSurfaceProperties(GPropertyMap* pstd, GPropertyMap* pmap, const char* prefix);
      GPropertyD* getPropertyOrDefault(GPropertyMap* pmap, const char* pname);

  private:
      // support for standardization 
      void defineDefaults(GPropertyMap* defaults);
      void setDefaults(GPropertyMap* defaults);
      GPropertyMap* getDefaults();
      GProperty<double>* getDefaultProperty(const char* name);
      GProperty<double>* getRamp();
      void setKeyMap(const char* spec);
      char*  digest(std::vector<GPropertyD*>& props);

  public:
      // another classes need access to "shape" of the standardization
      GDomain<double>*        getStandardDomain();
      static GDomain<double>* getDefaultDomain();
      unsigned int            getStandardDomainLength();

  public:
      void setMetadata(GSubstanceLibMetadata* meta); 

  public:
      void          dumpWavelengthBuffer(GBuffer* buffer);
      static void   dumpWavelengthBuffer(GBuffer* buffer, unsigned int numSubstance, unsigned int numProp, unsigned int domainLength);


#ifdef TRANSITIONAL
      GSubstanceLib* createStandardizedLib();
      void add(GSubstance* substance);
      void setStandard(bool standard);
      bool isStandard();
#endif
#ifdef LEGACY
      GPropertyMap* createStandardProperties(const char* name, GSubstance* substance);
      void checkMaterialProperties(GPropertyMap* ptex, unsigned int offset, const char* prefix);
      void checkSurfaceProperties(GPropertyMap* ptex, unsigned int offset, const char* prefix);
#endif

  public:
  private:
      std::map<std::string, std::string> m_keymap ; //  
      std::map<std::string, GSubstance*> m_registry ; 
      std::vector<std::string> m_keys ; 

      bool m_standard ; // transitional : keeping this set to true
      GDomain<double>* m_standard_domain ;  
      GPropertyMap* m_defaults ;  
      GProperty<double>* m_ramp ;  
      GSubstanceLibMetadata* m_meta ;

};


inline GSubstanceLibMetadata* GSubstanceLib::getMetadata()
{
    return m_meta ; 
}
inline void GSubstanceLib::setMetadata(GSubstanceLibMetadata* meta)
{
    m_meta = meta ; 
}
#ifdef TRANSITIONAL
inline void GSubstanceLib::setStandard(bool standard)
{
    m_standard = standard ;
}
inline bool GSubstanceLib::isStandard()
{
    return m_standard ;
}
#endif
inline void GSubstanceLib::setDefaults(GPropertyMap* defaults)
{
    m_defaults = defaults ;
}
inline GPropertyMap* GSubstanceLib::getDefaults()
{
    return m_defaults ;
}
inline GProperty<double>* GSubstanceLib::getRamp()
{
   return m_ramp ;
}
inline GDomain<double>* GSubstanceLib::getStandardDomain()
{
    return m_standard_domain ;
}



#endif
