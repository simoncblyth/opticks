#ifndef GMATERIAL_H
#define GMATERIAL_H

//
// analog of chroma.geometry.Material
//
//     refractive_index
//     absorption_length
//     scattering_length
//     reemission_prob
//     reemission_cdf
//

#include "GPropertyMap.hh"

class GMaterial : public GPropertyMap {
  public:
      GMaterial(const char* name);
      virtual ~GMaterial();

};


#endif


