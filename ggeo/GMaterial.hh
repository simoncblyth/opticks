#pragma once

//
// analog of chroma.geometry.Material
//
//     refractive_index
//     absorption_length
//     scattering_length
//     reemission_prob
//

#include "GPropertyMap.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

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


