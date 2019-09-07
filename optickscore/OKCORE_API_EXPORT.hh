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

Source "Generated" hdr OKCORE_API_EXPORT.hh 
Created Sun, Jun 19, 2016  5:55:49 PM with commandline::

    importlib-exports OpticksCore OKCORE_API  

https://cmake.org/Wiki/BuildingWinDLL

CMake will define OpticksCore_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define OpticksCore_EXPORTS when
building a DLL on windows.

*/

// TODO: probably mingw32 will need handling 

#if defined (_WIN32) 

   #if defined(OpticksCore_EXPORTS)
       #define  OKCORE_API __declspec(dllexport)
   #else
       #define  OKCORE_API __declspec(dllimport)
   #endif

#else

   #define OKCORE_API  __attribute__ ((visibility ("default")))

#endif


