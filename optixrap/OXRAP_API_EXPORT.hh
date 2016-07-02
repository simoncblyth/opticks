
#pragma once

/* 

Source "Generated" hdr OXRAP_API_EXPORT.hh 
Created Fri Jun 24 20:52:04 CST 2016 with commandline::

    importlib-exports OptiXRap OXRAP_API  

https://cmake.org/Wiki/BuildingWinDLL

CMake will define OptiXRap_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define OptiXRap_EXPORTS when
building a DLL on windows.

*/

// TODO: probably mingw32 will need handling 

#if defined (_WIN32) 

   #if defined(OptiXRap_EXPORTS)
       #define  OXRAP_API __declspec(dllexport)
   #else
       #define  OXRAP_API __declspec(dllimport)
   #endif

#else

   #define OXRAP_API  __attribute__ ((visibility ("default")))

#endif


