
#pragma once

/* 

Source "Generated" hdr YOG_API_EXPORT.hh 
Created Sat Jun  2 15:25:03 HKT 2018 with commandline::

    winimportlib-exports YoctoGLRap YOG_API  

https://cmake.org/Wiki/BuildingWinDLL

CMake will define YoctoGLRap_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define YoctoGLRap_EXPORTS when
building a DLL on windows.

*/

// TODO: probably mingw32 will need handling 

#if defined (_WIN32) 

   #if defined(YoctoGLRap_EXPORTS)
       #define  YOG_API __declspec(dllexport)
   #else
       #define  YOG_API __declspec(dllimport)
   #endif

#else

   #define YOG_API  __attribute__ ((visibility ("default")))

#endif


