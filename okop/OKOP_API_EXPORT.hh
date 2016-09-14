
#pragma once

/* 

Source "Generated" hdr OKOP_API_EXPORT.hh 
Created Sat Jun 25 13:29:29 CST 2016 with commandline::

    importlib-exports OKOP OKOP_API  

https://cmake.org/Wiki/BuildingWinDLL

CMake will define OKOP_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define OKOP_EXPORTS when
building a DLL on windows.

*/

// TODO: probably mingw32 will need handling 

#if defined (_WIN32) 

   #if defined(OKOP_EXPORTS)
       #define  OKOP_API __declspec(dllexport)
   #else
       #define  OKOP_API __declspec(dllimport)
   #endif

#else

   #define OKOP_API  __attribute__ ((visibility ("default")))

#endif


