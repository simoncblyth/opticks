#ifndef GSURFACE_H
#define GSURFACE_H

//
// analog of chroma.geometry.Surface
//
//    detect
//    absorb
//    reemit
//    reflect_diffuse
//    reflect_specular
//    eta
//    k
//    reemission_cdf
//

#include "GPropertyMap.hh"

class GSurface : public GPropertyMap {
  public:
      GSurface(const char* name);
      virtual ~GSurface();

};

#endif


