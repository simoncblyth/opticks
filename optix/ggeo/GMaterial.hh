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

class GMaterial : public GPropertyMap<float> {
  public:
      GMaterial(GMaterial* other);
      GMaterial(const char* name, unsigned int index);
      virtual ~GMaterial();
  private:
      void init(); 
  public: 
      void Summary(const char* msg="GMaterial::Summary");

};




