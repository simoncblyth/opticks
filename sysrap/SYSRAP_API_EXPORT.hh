
#pragma once

/* 

Source "Generated" hdr SYSRAP_API_EXPORT.hh 
Created Fri Jun 24 18:50:00 CST 2016 with commandline::

    importlib-exports SysRap SYSRAP_API  

https://cmake.org/Wiki/BuildingWinDLL

CMake will define SysRap_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define SysRap_EXPORTS when
building a DLL on windows.

*/

// TODO: probably mingw32 will need handling 

#if defined (_WIN32) 

   #if defined(SysRap_EXPORTS)
       #define  SYSRAP_API __declspec(dllexport)
   #else
       #define  SYSRAP_API __declspec(dllimport)
   #endif

#else

   #define SYSRAP_API  __attribute__ ((visibility ("default")))

#endif


