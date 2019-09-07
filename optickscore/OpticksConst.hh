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

#include <string>
#include "OKCORE_API_EXPORT.hh"

class OKCORE_API OpticksConst {
   public:
       static const char* BNDIDX_NAME_ ;
       static const char* SEQHIS_NAME_ ;
       static const char* SEQMAT_NAME_ ;
   public:
       enum { 
              e_shift   = 1 << 0,  
              e_control = 1 << 1,  
              e_option  = 1 << 2,  
              e_command = 1 << 3 
            } ; 
      // GLFW_MOD_SHIFT
      // GLFW_MOD_CONTROL
      // GLFW_MOD_ALT
      // GLFW_MOD_SUPER

       static bool isShift(unsigned int modifiers);
       static bool isControl(unsigned int modifiers);
       static bool isShiftOption(unsigned int modifiers);
       static bool isCommand(unsigned int modifiers);  // <-- Linux super key often redefined, unlike macOS command key so not so useful
       static bool isOption(unsigned int modifiers);
       static std::string describeModifiers(unsigned int modifiers);
   public:
       static const char GEOCODE_ANALYTIC ;
       static const char GEOCODE_TRIANGULATED ;
       static const char GEOCODE_GEOMETRYTRIANGLES ;
       static const char GEOCODE_SKIP  ;

};




 
