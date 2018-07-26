
#pragma once

/* 

Source "Generated" hdr Y4CSG_API_EXPORT.hh 
Created Thu Jul 26 11:20:53 CST 2018 with commandline::

    winimportlib-exports Y4CSG Y4CSG_API  

https://cmake.org/Wiki/BuildingWinDLL

CMake will define Y4CSG_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define Y4CSG_EXPORTS when
building a DLL on windows.

*/

// TODO: probably mingw32 will need handling 

#if defined (_WIN32) 

   #if defined(Y4CSG_EXPORTS)
       #define  Y4CSG_API __declspec(dllexport)
   #else
       #define  Y4CSG_API __declspec(dllimport)
   #endif

#else

   #define Y4CSG_API  __attribute__ ((visibility ("default")))

#endif


