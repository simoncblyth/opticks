#pragma once

#include "GProperty.hh"
#include "GDomain.hh"

#include <string>
#include <vector>
#include <map>


template <class T>
class GPropertyMap {

  typedef std::map<std::string,GProperty<T>*> GPropertyMap_t ;
  public:
      GPropertyMap(GPropertyMap* other);
      GPropertyMap(const char* name);
      GPropertyMap(const char* name, unsigned int index, const char* type);
      virtual ~GPropertyMap();

  public:
     // caller should free the char* returned after dumping 
      //char* digest();
      char* pdigest(int ifr, int ito); 
      char* getShortName(const char* prefix); 
      std::string getPDigestString(int ifr, int ito);
      std::string getShortNameString(const char* prefix="__dd__Materials__");
      std::string getKeysString(); 
      std::string description();

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
      void setStandardDomain(GDomain<T>* standard_domain);
      GDomain<T>* getStandardDomain();

  public:
      void add(GPropertyMap<T>* other, const char* prefix=NULL);
      void addConstantProperty(const char* pname, T value, const char* prefix=NULL);
      void addProperty(const char* pname, T* values, T* domain, unsigned int length, const char* prefix=NULL);
      void addProperty(const char* pname, GProperty<T>* prop, const char* prefix=NULL);
      unsigned int getNumProperties();

  public:
      GProperty<T>* getPropertyByIndex(int index);
      const char* getPropertyNameByIndex(int index);
      GProperty<T>* getProperty(const char* pname);
      std::vector<std::string>& getKeys();

  private:
      std::string m_name ;
      std::string m_shortname ; 
      std::string m_type ;
      unsigned int m_index ;

      GPropertyMap_t m_prop ; 
      std::vector<std::string> m_keys ;  // key ordering

      GDomain<T>* m_standard_domain ; 

};




