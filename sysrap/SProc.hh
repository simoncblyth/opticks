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

Almost all in dead code::

    epsilon:opticks blyth$ opticks-fl SProc.hh

    ./sysrap/SProc.hh
    ./sysrap/CMakeLists.txt
    ./sysrap/tests/SProcTest.cc
    ./sysrap/SProc.cc

    ./cfg4/tests/CGenstepCollectorLeak2Test.cc
    ./cfg4/tests/CGenstepCollectorLeakTest.cc
    ./optickscore/OpticksProfile.cc
    ./optickscore/tests/OpticksEventLeakTest.cc
    ./optickscore/tests/OpticksRunTest.cc
    ./optickscore/tests/OpticksEventTest.cc
    ./optickscore/Opticks.cc
    ./boostrap/BOpticksResource.cc
    ./npy/tests/NPY8DynamicRandomTest.cc
    ./npy/tests/NPY5Test.cc
    ./npy/tests/NPY6Test.cc

But profiling usage is currently all in dead code::

    epsilon:opticks blyth$ opticks-fl VirtualMemoryUsage
    ./cfg4/tests/CGenstepCollectorLeak2Test.cc
    ./cfg4/tests/CGenstepCollectorLeakTest.cc
    ./sysrap/sproc.h
    ./sysrap/SProc.hh
    ./sysrap/tests/reallocTest.cc
    ./sysrap/tests/SProcTest.cc
    ./sysrap/SProc.cc
    ./optickscore/OpticksProfile.cc
    ./optickscore/tests/OpticksRunTest.cc
    ./optickscore/tests/OpticksEventTest.cc
    ./npy/tests/NPY5Test.cc
    ./npy/tests/NPY6Test.cc
    epsilon:opticks blyth$ 

    epsilon:opticks blyth$ opticks-fl ResidentSetSize 
    ./cfg4/tests/CGenstepCollectorLeak2Test.cc
    ./sysrap/sproc.h
    ./sysrap/SProc.hh
    ./sysrap/SProc.cc
    ./optickscore/OpticksProfile.cc
    ./npy/tests/NPY6Test.cc
    epsilon:opticks blyth$ 


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
