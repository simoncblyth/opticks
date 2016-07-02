
#pragma once

/* 

Source "Generated" hdr CUDARAP_API_EXPORT.hh 
Created Fri Jun 24 18:09:13 CST 2016 with commandline::

    importlib-exports CUDARap CUDARAP_API  

https://cmake.org/Wiki/BuildingWinDLL

CMake will define CUDARap_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define CUDARap_EXPORTS when
building a DLL on windows.

*/

// TODO: probably mingw32 will need handling 

#if defined (_WIN32) 

   #if defined(CUDARap_EXPORTS)
       #define  CUDARAP_API __declspec(dllexport)
   #else
       #define  CUDARAP_API __declspec(dllimport)
   #endif

#else

   #define CUDARAP_API  __attribute__ ((visibility ("default")))

#endif


