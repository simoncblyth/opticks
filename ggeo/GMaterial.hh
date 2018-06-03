#pragma once

#include "GPropertyMap.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

/**
GMaterial
===========

1. thin layer over base GPropertyMap<float> 


G4DAE Workflow
----------------

1. GMaterial instances created by AssimpGGeo::convertMaterials 
2. populated by AssimpGGeo::addProperties(GPropertyMap<float>* pmap, aiMaterial* material )
   where G4 property vectors are pulled out of the assimp materials 

**/

class GGEO_API GMaterial : public GPropertyMap<float> {
  public:
      GMaterial(GMaterial* other, GDomain<float>* domain = NULL);  // non-NULL domain interpolates
      GMaterial(const char* name, unsigned int index);
      virtual ~GMaterial();
  private:
      void init(); 
  public: 
      void Summary(const char* msg="GMaterial::Summary");

};

#include "GGEO_TAIL.hh"


