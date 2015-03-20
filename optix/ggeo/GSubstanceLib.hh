#ifndef GSUBSTANCELIB_H
#define GSUBSTANCELIB_H

#include <map>
#include <vector>
#include <string>

#include "GDomain.hh"
#include "GProperty.hh"

class GSubstance ; 
class GPropertyMap ; 

class GSubstanceLib {
  public:
    // standard property prefixes
    static const char* inner; 
    static const char* outer; 
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
      GSubstance* get(GPropertyMap* imaterial, GPropertyMap* omaterial, GPropertyMap* isurface, GPropertyMap* osurface );

  public:
      GSubstanceLib();
      virtual ~GSubstanceLib();

  public:
      void setStandardDomain(GDomain<double>* standard_domain);
      GDomain<double>* getStandardDomain();

  public:
      GPropertyMap* createStandardProperties(const char* name, GSubstance* substance);

  private:
      void addMaterialProperties(GPropertyMap* ptex, GPropertyMap* pmap, const char* prefix);
      void addSurfaceProperties(GPropertyMap* ptex, GPropertyMap* pmap, const char* prefix);
      void checkMaterialProperties(GPropertyMap* ptex, unsigned int offset, const char* prefix);
      void checkSurfaceProperties(GPropertyMap* ptex, unsigned int offset, const char* prefix);
      GPropertyD* getPropertyOrDefault(GPropertyMap* pmap, const char* pname);

  public:
      void setKeyMap(const char* spec);
      const char* getLocalKey(const char* dkey); // map standard -> local keys 

  public:
      void defineDefaults(GPropertyMap* defaults);
      void setDefaults(GPropertyMap* defaults);
      GPropertyMap* getDefaults();
      GProperty<double>* getDefaultProperty(const char* name);
      GProperty<double>* getRamp();

  public:
      unsigned int getNumSubstances();
      GSubstance* getSubstance(unsigned int index); 

  public:
      void Summary(const char* msg="GSubstanceLib::Summary");

  private:
      std::map<std::string, std::string> m_keymap ; 
      std::map<std::string, GSubstance*> m_registry ; 
      std::vector<std::string> m_keys ; 

      GDomain<double>* m_standard_domain ;  
      GPropertyMap* m_defaults ;  
      GProperty<double>* m_ramp ;  

};



#endif
