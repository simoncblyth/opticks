#ifndef GPROPERTYMAP_H
#define GPROPERTYMAP_H

#include "GProperty.hh"
#include <string>
#include <map>


class GPropertyMap {
  public:
      GPropertyMap(const char* name);
      virtual ~GPropertyMap();
  public:
      void AddProperty(const char* pname, float* values, float* domain, size_t length );
      GPropertyF* GetProperty(const char* pname);

  private:
      std::string m_name ;
      std::map<std::string,GPropertyF*> m_prop ; 
};


#endif


