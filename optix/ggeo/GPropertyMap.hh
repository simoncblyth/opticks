#pragma once

#include "GProperty.hh"
#include "GDomain.hh"

class GOpticalSurface ; 

#include <string>
#include <vector>
#include <map>


template <class T>
class GPropertyMap {

  static const char* NOT_DEFINED ;
  typedef std::map<std::string,GProperty<T>*> GPropertyMap_t ;
  public:
      GPropertyMap(GPropertyMap* other);
      GPropertyMap(const char* name);
      GPropertyMap(const char* name, unsigned int index, const char* type, GOpticalSurface* optical_surface=NULL);
      virtual ~GPropertyMap();

  public:
     // caller should free the char* returned after dumping 
      //char* ndigest();
      char* pdigest(int ifr, int ito); 
      const char* getShortName() const ; 
      bool hasShortName(const char* name);
      bool hasDefinedName();
      bool hasNameEnding(const char* end);

      std::string getShortNameString();
      std::string getPDigestString(int ifr, int ito);
      std::string getKeysString(); 
      std::string description();

  private:
      void findShortName(const char* prefix="__dd__Materials__");
      //char* trimSuffixPrefix(const char* origname, const char* prefix=NULL);

  public:
      const char* getName();    // names like __dd__Materials__Nylon0xc3aa360 or __dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear2
      unsigned int getIndex();  // aiScene material index ("surfaces" and "materials" represented as Assimp materials)
      const char* getType();

      bool isSurface();
      bool isSkinSurface();
      bool isBorderSurface();
      bool isMaterial();

      void setSensor(bool sensor=true); // set in AssimpGGeo::convertSensors
      bool isSensor();

      void setValid(bool valid=true);
      bool isValid();

      void setOpticalSurface(GOpticalSurface* optical_surface);
      GOpticalSurface* getOpticalSurface(); 


      void dump(const char* msg="GPropertyMap::Summary", unsigned int nline=1);
      void Summary(const char* msg="GPropertyMap::Summary", unsigned int nline=1);

  public:
      bool hasStandardDomain();
      void setStandardDomain(GDomain<T>* standard_domain);
      GDomain<T>* getStandardDomain();

  public:
      void add(GPropertyMap<T>* other, const char* prefix=NULL);
      void addConstantProperty(const char* pname, T value, const char* prefix=NULL);

      // when a standard domain is defined these methods interpolates the values provided onto that domain
      void addProperty(const char* pname, T* values, T* domain, unsigned int length, const char* prefix=NULL);
      void addPropertyStandardized(const char* pname,  GProperty<T>* orig, const char* prefix=NULL);

       // this one does not interpolate  
      void addProperty(const char* pname, GProperty<T>* prop, const char* prefix=NULL);

      unsigned int getNumProperties() const ;

  public:
      GProperty<T>* getPropertyByIndex(int index) ;
      const char* getPropertyNameByIndex(int index) ;
      GProperty<T>* getProperty(const char* pname)  ;
      GProperty<T>* getProperty(const char* pname, const char* prefix);
      bool hasProperty(const char* pname) ;
      std::vector<std::string>& getKeys() ;

  private:
      std::string m_name ;
      const char* m_shortname ; 
      std::string m_type ;

      unsigned int m_index ;
      bool         m_sensor ;  
      bool         m_valid ;  

      GPropertyMap_t m_prop ; 
      std::vector<std::string> m_keys ;  // key ordering

      GDomain<T>* m_standard_domain ; 
      GOpticalSurface* m_optical_surface ; 

};


template <class T>
inline void GPropertyMap<T>::setOpticalSurface(GOpticalSurface* optical_surface)
{
    m_optical_surface = optical_surface ;
}
template <class T>
inline GOpticalSurface* GPropertyMap<T>::getOpticalSurface()
{
    return m_optical_surface ; 
}
template <class T>
inline bool GPropertyMap<T>::hasNameEnding(const char* end)
{
    if(m_name.length() < strlen(end))
    {
        return false ; 
    }
    else
    {
        return 0 == m_name.compare( m_name.length() - strlen(end), strlen(end), end ) ;
    }
}


template <class T>
inline bool GPropertyMap<T>::isSensor()
{
    return m_sensor ; 
}
template <class T>
inline void GPropertyMap<T>::setSensor(bool sensor)
{
    m_sensor = sensor ; 
}

template <class T>
inline bool GPropertyMap<T>::isValid()
{
    return m_valid ; 
}
template <class T>
inline void GPropertyMap<T>::setValid(bool valid)
{
    m_valid = valid ; 
}




