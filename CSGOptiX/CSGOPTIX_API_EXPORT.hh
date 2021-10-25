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

/* 

Source "Generated" hdr CSGOPTIX_API_EXPORT.hh 
Created Fri, Jun 17, 2016  8:03:44 PM with commandline::

    importlib-exports CSGOPTIX CSGOPTIX_API  

https://cmake.org/Wiki/BuildingWinDLL

CMake will define CSGOPTIX_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define CSGOPTIX_EXPORTS when
building a DLL on windows.

*/

// TODO: probably mingw32 will need handling 

#if defined (_WIN32) 

   #if defined(CSGOPTIX_EXPORTS)
       #define  CSGOPTIX_API __declspec(dllexport)
   #else
       #define  CSGOPTIX_API __declspec(dllimport)
   #endif

#else

   #define CSGOPTIX_API  __attribute__ ((visibility ("default")))

#endif


