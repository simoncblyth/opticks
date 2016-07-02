
#pragma once

/* 

Source "Generated" hdr MESHRAP_API_EXPORT.hh 
Created Thu, Jun 23, 2016  5:00:14 PM with commandline::

    importlib-exports OpenMeshRap MESHRAP_API  

https://cmake.org/Wiki/BuildingWinDLL

CMake will define OpenMeshRap_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define OpenMeshRap_EXPORTS when
building a DLL on windows.

*/

// TODO: probably mingw32 will need handling 

#if defined (_WIN32) 

   #if defined(OpenMeshRap_EXPORTS)
       #define  MESHRAP_API __declspec(dllexport)
   #else
       #define  MESHRAP_API __declspec(dllimport)
   #endif

#else

   #define MESHRAP_API  __attribute__ ((visibility ("default")))

#endif


