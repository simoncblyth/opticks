
#pragma once

/* 

Source "Generated" hdr OKG4_API_EXPORT.hh 
Created Tue Aug 30 20:12:33 CST 2016 with commandline::

    importlib-exports okg4 OKG4_API  

https://cmake.org/Wiki/BuildingWinDLL

CMake will define okg4_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define okg4_EXPORTS when
building a DLL on windows.

*/

// TODO: probably mingw32 will need handling 

#if defined (_WIN32) 

   #if defined(okg4_EXPORTS)
       #define  OKG4_API __declspec(dllexport)
   #else
       #define  OKG4_API __declspec(dllimport)
   #endif

#else

   #define OKG4_API  __attribute__ ((visibility ("default")))

#endif


