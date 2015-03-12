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
      GSubstance* get(GPropertyMap* imaterial, GPropertyMap* omaterial, GPropertyMap* isurface, GPropertyMap* osurface );

  public:
      GSubstanceLib();
      virtual ~GSubstanceLib();

  public:
      void setStandardDomain(GDomain<double>* standard_domain);
      GDomain<double>* getStandardDomain();

  public:
      void setDefaults(GPropertyMap* defaults);
      GPropertyMap* getDefaults();
      GProperty<double>* getDefaultProperty(const char* name);

  public:
      unsigned int getNumSubstances();
      GSubstance* getSubstance(unsigned int index); 

  public:
      void Summary(const char* msg="GSubstanceLib::Summary");

  private:
      std::map<std::string, GSubstance*> m_registry ; 
      std::vector<std::string> m_keys ; 

      GDomain<double>* m_standard_domain ;  
      GPropertyMap* m_defaults ;  

};



#endif
