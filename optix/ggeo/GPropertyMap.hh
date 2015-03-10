#ifndef GPROPERTYMAP_H
#define GPROPERTYMAP_H

#include "GProperty.hh"
#include <string>
#include <map>


class GPropertyMap {

  typedef std::map<std::string,GPropertyF*> GPropertyMapF_t ;
  public:
      GPropertyMap(const char* name, unsigned int index, const char* type);
      virtual ~GPropertyMap();

  public:
     // caller should free the char* returned after dumping 
      char* digest();
      char* getShortName(const char* prefix); 
      char* getKeys(); 

  public:
      const char* getName();
      unsigned int getIndex();
      const char* getType();

      bool isSkinSurface();
      bool isBorderSurface();
      bool isMaterial();

      void Summary(const char* msg="GPropertyMap::Summary");

  public:
      void addProperty(const char* pname, float* values, float* domain, size_t length );
      GPropertyF* getProperty(const char* pname);
      void setStandardDomain( float low, float high, float step);
      float getLow();
      float getHigh();
      float getStep();


  private:
      std::string m_name ;
      unsigned int m_index ;
      std::string m_type ;
      GPropertyMapF_t m_prop ; 

  private:
      float m_low ;
      float m_high ;
      float m_step ;


};


#endif


