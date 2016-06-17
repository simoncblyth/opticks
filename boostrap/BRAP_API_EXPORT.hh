
#pragma once

/* 

Source "Generated" hdr BRAP_API_EXPORT.hh 
Created Thu, Jun 16, 2016 11:42:52 AM with commandline::

    importlib-exports BoostRap BRAP_API  

https://cmake.org/Wiki/BuildingWinDLL

CMake will define BoostRap_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define BoostRap_EXPORTS when
building a DLL on windows.

*/

// TODO: probably mingw32 will need handling 

#ifdef _WIN32 

   #ifdef BoostRap_EXPORTS
       #define  BRAP_API __declspec(dllexport)
   #else
       #define  BRAP_API __declspec(dllimport)
   #endif

#else

   #define BRAP_API

#endif





