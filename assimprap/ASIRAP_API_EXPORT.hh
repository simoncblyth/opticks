
#pragma once

/* 

Source "Generated" hdr ASIRAP_API_EXPORT.hh 
Created Thu, Jun 23, 2016  3:34:59 PM with commandline::

    importlib-exports AssimpRap ASIRAP_API  

https://cmake.org/Wiki/BuildingWinDLL

CMake will define AssimpRap_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define AssimpRap_EXPORTS when
building a DLL on windows.

*/

// TODO: probably mingw32 will need handling 

#if defined (_WIN32) 

   #if defined(AssimpRap_EXPORTS)
       #define  ASIRAP_API __declspec(dllexport)
   #else
       #define  ASIRAP_API __declspec(dllimport)
   #endif

#else

   #define ASIRAP_API  __attribute__ ((visibility ("default")))

#endif


