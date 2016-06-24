
#pragma once

/* 

Source "Generated" hdr GGV_API_EXPORT.hh 
Created Fri, Jun 24, 2016  1:15:57 PM with commandline::

    importlib-exports GGeoView GGV_API  

https://cmake.org/Wiki/BuildingWinDLL

CMake will define GGeoView_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define GGeoView_EXPORTS when
building a DLL on windows.

*/

// TODO: probably mingw32 will need handling 

#if defined (_WIN32) 

   #if defined(GGeoView_EXPORTS)
       #define  GGV_API __declspec(dllexport)
   #else
       #define  GGV_API __declspec(dllimport)
   #endif

#else

   #define GGV_API  __attribute__ ((visibility ("default")))

#endif


