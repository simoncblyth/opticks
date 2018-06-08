#pragma once

#include <string>
#include <vector>
#include <map>


template <typename T> class GPropertyMap ; 
template <typename T> class GProperty ; 
template <typename T> class GDomain ; 

class NMeta ; 
class GOpticalSurface ; 

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

/**

GPropertyMap<T>
==================

1. manages m_prop : a string keyed map of GProperty<T>


TODO: const correctness would be good, although painful to implement

**/

template <class T>
class GGEO_API GPropertyMap {

  static const char* NOT_DEFINED ;
  typedef std::map<std::string,GProperty<T>*> GPropertyMap_t ;
  public:
      GPropertyMap(GPropertyMap* other, GDomain<T>* domain=NULL);  // used for interpolation when domain provided
      GPropertyMap(const char* name);
      GPropertyMap(const char* name, unsigned int index, const char* type, GOpticalSurface* optical_surface=NULL, NMeta* meta=NULL);

      virtual ~GPropertyMap();
  private:
      void init();
      void collectMeta();
  public:
      void dumpMeta(const char* msg="GPropertyMap::dumpMeta") const ;
      void save(const char* path);
      static GPropertyMap<T>* load(const char* path, const char* name, const char* type);
  public:
     // caller should free the char* returned after dumping 
      //char* ndigest();
      char* pdigest(int ifr, int ito); 
      const char* getShortName() const ; 
      bool hasShortName(const char* name);
      bool hasDefinedName();
      bool hasNameEnding(const char* end);

      NMeta* getMeta() const ; 
      std::string getMetaDesc() const ; 

      template <typename S> 
      void setMetaKV(const char* key, S value);
      bool hasMetaItem(const char* key ) const ;


      std::string getShortNameString();
      std::string getPDigestString(int ifr, int ito);
      std::string getKeysString(); 
      std::string description();
  public:
      std::string make_table(unsigned int fwid=20, T dscale=1, bool dreciprocal=false);
  public:
      GPropertyMap<T>* spawn_interpolated(T nm=1.0f);
  public:
      static const char* FindShortName(const char* name, const char* prefix);
  private:
      void findShortName(const char* prefix="__dd__Materials__");
      //char* trimSuffixPrefix(const char* origname, const char* prefix=NULL);

  public:
      std::string brief() const ; 
      const char* getName();    // names like __dd__Materials__Nylon0xc3aa360 or __dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear2
      unsigned getIndex() const ;  // aiScene material index ("surfaces" and "materials" represented as Assimp materials)
      void setIndex(unsigned index);

      const char* getType();


      void setSkinSurface(); 
      void setBorderSurface(); 


      bool isSurface() const ;
      bool isTestSurface() const ;
      bool isSkinSurface() const ;
      bool isBorderSurface() const ;
      bool isMaterial() const ;
      bool hasNonZeroProperty(const char* pname) ;
     
   public:
      // from metadata
      std::string getBPV1() const ; 
      std::string getBPV2() const ; 
      std::string getSSLV() const ; 
  public:

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
      T getDomainLow();
      T getDomainHigh();
      T getDomainStep();
  public:
      void add(GPropertyMap<T>* other, const char* prefix=NULL);
      void addStandardized(GPropertyMap<T>* other, const char* prefix=NULL);
      void addConstantProperty(const char* pname, T value, const char* prefix=NULL);
      bool setPropertyValues(const char* pname, T val); 

      // when a standard domain is defined these methods interpolates the values provided onto that domain
      void addProperty(const char* pname, T* values, T* domain, unsigned int length, const char* prefix=NULL);
      void addPropertyStandardized(const char* pname,  GProperty<T>* orig, const char* prefix=NULL);

       // this one does not interpolate  
      void addProperty(const char* pname, GProperty<T>* prop, const char* prefix=NULL);
      void replaceProperty(const char* pname, GProperty<T>* prop, const char* prefix=NULL);

      unsigned int getNumProperties() const ;


  public:
      GProperty<T>* getPropertyByIndex(int index) ;
      const char* getPropertyNameByIndex(int index) ;
      GProperty<T>* getProperty(const char* pname)   ;
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

      GPropertyMap_t           m_prop ; 
      std::vector<std::string> m_keys ;  // key ordering

      GDomain<T>*      m_standard_domain ; 
      GOpticalSurface* m_optical_surface ; 

      NMeta*      m_meta ; 


};


#include "GGEO_TAIL.hh"

