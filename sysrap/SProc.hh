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

/**
SProc : MIGRATING to sproc.h
=================================

::

    epsilon:sysrap blyth$ opticks-f SProc.hh 
    ## ./CSGOptiX/CSGOptiX.cc:#include "SProc.hh"
    ## ./CSG/CSGFoundry.cc:#include "SProc.hh"
    ## ./sysrap/SProc.cc:#include "SProc.hh"
    ## ./sysrap/SLOG.cc:#include "SProc.hh"
    ## ./sysrap/CMakeLists.txt:    SProc.hh

    ./sysrap/tests/SProcTest.cc:#include "SProc.hh"

    ./sysrap/SGeo.cc:#include "SProc.hh"

    ./sysrap/SOpticks.cc:#include "SProc.hh"
    ./sysrap/SOpticksResource.cc:#include "SProc.hh"





**/

#include <cstddef>

#include "SYSRAP_API_EXPORT.hh"

class SYSRAP_API SProc {
  public:
      static float VirtualMemoryUsageMB();
      static float VirtualMemoryUsageKB();
      static float ResidentSetSizeMB();
      static float ResidentSetSizeKB();


      static const char* ExecutablePath(bool basename=false); 
      static const char* ExecutableName(); 


};
