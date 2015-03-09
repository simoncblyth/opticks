#ifndef GSUBSTANCELIB_H
#define GSUBSTANCELIB_H

#include <map>
#include <vector>
#include <string>

class GSubstance ; 
class GPropertyMap ; 


class GSubstanceLib {

  public:
      GSubstance* get(GPropertyMap* imaterial, GPropertyMap* isurface, GPropertyMap* osurface );

  public:
      GSubstanceLib();
      virtual ~GSubstanceLib();

  public:
      void Summary(const char* msg="GSubstanceLib::Summary");

  private:
      std::map<std::string, GSubstance*> m_registry ; 
      std::vector<std::string> m_keys ; 

};



#endif
