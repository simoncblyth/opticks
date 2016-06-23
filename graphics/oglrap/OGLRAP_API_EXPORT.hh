
#pragma once

/* 

Source "Generated" hdr OGLRAP_API_EXPORT.hh 
Created Thu, Jun 23, 2016  6:59:23 PM with commandline::

    importlib-exports OGLRap OGLRAP_API  

https://cmake.org/Wiki/BuildingWinDLL

CMake will define OGLRap_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define OGLRap_EXPORTS when
building a DLL on windows.

*/

// TODO: probably mingw32 will need handling 

#if defined (_WIN32) 

   #if defined(OGLRap_EXPORTS)
       #define  OGLRAP_API __declspec(dllexport)
   #else
       #define  OGLRAP_API __declspec(dllimport)
   #endif

#else

   #define OGLRAP_API  __attribute__ ((visibility ("default")))

#endif


