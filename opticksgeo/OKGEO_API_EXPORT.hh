
#pragma once

/* 

Source "Generated" hdr OKGEO_API_EXPORT.hh 
Created Thu, Jun 23, 2016  6:04:51 PM with commandline::

    importlib-exports OpticksGeometry OKGEO_API  

https://cmake.org/Wiki/BuildingWinDLL

CMake will define OpticksGeometry_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define OpticksGeometry_EXPORTS when
building a DLL on windows.

*/

// TODO: probably mingw32 will need handling 

#if defined (_WIN32) 

   #if defined(OpticksGeometry_EXPORTS)
       #define  OKGEO_API __declspec(dllexport)
   #else
       #define  OKGEO_API __declspec(dllimport)
   #endif

#else

   #define OKGEO_API  __attribute__ ((visibility ("default")))

#endif


