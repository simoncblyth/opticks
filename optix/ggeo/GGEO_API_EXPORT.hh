
#pragma once

/* 
Source "Generated" hdr GGEO_API_EXPORT.hh 
Created Tue, Jun 21, 2016  8:12:47 PM with commandline::

    importlib-exports GGeo GGEO_API  

https://cmake.org/Wiki/BuildingWinDLL

CMake will define GGeo_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define GGeo_EXPORTS when
building a DLL on windows.

*/

// TODO: probably mingw32 will need handling 

#if defined (_WIN32) 

   #if defined(GGeo_EXPORTS)
       #define  GGEO_API __declspec(dllexport)
   #else
       #define  GGEO_API __declspec(dllimport)
   #endif

#else

   #define GGEO_API  __attribute__ ((visibility ("default")))

#endif


