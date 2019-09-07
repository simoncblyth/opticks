/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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

