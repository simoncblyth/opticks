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

#include "C4FPEDetection.hh"
#include <iostream>


#if (defined(__GNUC__) && !defined(__clang__))
#ifdef __linux__
  //#include <features.h>
  #include <fenv.h>
  //#include <csignal>

void C4FPEDetection::InvalidOperationDetection_Disable()
{

/*
      std::cout 
              << std::endl
              << "        "
              << "C4FPEDetection::InvalidOperationDetection_Disable"
              << std::endl
              << "        "
              << "############################################" << std::endl
              << "        "
              << "!!! WARNING - FPE detection is DISABLED  !!!" << std::endl
              << "        "
              << "############################################" << std::endl
              << std::endl
              ; 

*/

    (void) fedisableexcept( FE_DIVBYZERO );
    (void) fedisableexcept( FE_INVALID );

}

#endif

#else

void C4FPEDetection::InvalidOperationDetection_Disable()
{
      std::cout 
              << std::endl
              << "  C4FPEDetection::InvalidOperationDetection_Disable      "
              << " NOT IMPLEMENTED "
              << std::endl
              ;
}

#endif


