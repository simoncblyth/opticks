#ifndef GSUBSTANCELIB_H
#define GSUBSTANCELIB_H

#include <vector>
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
      std::vector<GSubstance*> g_registry ; 

};



#endif
