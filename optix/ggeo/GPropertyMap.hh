#ifndef GPROPERTYMAP_H
#define GPROPERTYMAP_H

#include "GProperty.hh"
#include <string>
#include <vector>
#include <map>


class GPropertyMap {

  typedef std::map<std::string,GPropertyD*> GPropertyMapD_t ;
  public:
      GPropertyMap(const char* name);
      GPropertyMap(const char* name, unsigned int index, const char* type);
      virtual ~GPropertyMap();

  public:
     // caller should free the char* returned after dumping 
      char* digest();
      char* pdigest(int ifr, int ito); 
      char* getShortName(const char* prefix); 
      std::string getKeysString(); 

  public:
      const char* getName();    // names like __dd__Materials__Nylon0xc3aa360 or __dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear2
      unsigned int getIndex();  // aiScene material index ("surfaces" and "materials" represented as Assimp materials)
      const char* getType();

      bool isSkinSurface();
      bool isBorderSurface();
      bool isMaterial();

      void Summary(const char* msg="GPropertyMap::Summary", unsigned int nline=1);

  public:
      bool hasStandardDomain();
      void setStandardDomain(GDomain<double>* standard_domain);
      GDomain<double>* getStandardDomain();

  public:
      void addConstantProperty(const char* pname, double value, const char* prefix=NULL);
      void addProperty(const char* pname, double* values, double* domain, unsigned int length, const char* prefix=NULL);
      void addProperty(const char* pname, GPropertyD* prop, const char* prefix=NULL);
      unsigned int getNumProperties();

  public:
      GPropertyD* getPropertyByIndex(int index);
      const char* getPropertyNameByIndex(int index);
      GPropertyD* getProperty(const char* pname);
      std::vector<std::string>& getKeys();

  private:
      std::string m_name ;
      std::string m_type ;
      unsigned int m_index ;

      GPropertyMapD_t m_prop ; 
      std::vector<std::string> m_keys ;  // key ordering

      GDomain<double>* m_standard_domain ; 


};


#endif


