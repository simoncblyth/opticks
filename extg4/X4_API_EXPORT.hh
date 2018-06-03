
#pragma once

/* 

Source "Generated" hdr X4_API_EXPORT.hh 
Created Sun Jun  3 16:22:51 HKT 2018 with commandline::

    winimportlib-exports ExtG4 X4_API  

https://cmake.org/Wiki/BuildingWinDLL

CMake will define ExtG4_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define ExtG4_EXPORTS when
building a DLL on windows.

*/

// TODO: probably mingw32 will need handling 

#if defined (_WIN32) 

   #if defined(ExtG4_EXPORTS)
       #define  X4_API __declspec(dllexport)
   #else
       #define  X4_API __declspec(dllimport)
   #endif

#else

   #define X4_API  __attribute__ ((visibility ("default")))

#endif


