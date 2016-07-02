#pragma once

#include "NPY_API_EXPORT.hh"


/*
// this way gives no error but the 
// symbol fails to cross the dll divide 

NPY_API float cie_X(float wavelength);
*/


float NPY_API cie_X(float wavelength);
float NPY_API cie_Y(float wavelength);
float NPY_API cie_Z(float wavelength);


class NPY_API NCIE {
  public:
      static float X(float wavelength);
      static float Y(float wavelength);
      static float Z(float wavelength);
};

