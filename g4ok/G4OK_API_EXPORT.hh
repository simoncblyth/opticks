
#pragma once

/* 

Source "Generated" hdr G4OK_API_EXPORT.hh 
Created Mon May 28 14:01:27 HKT 2018 with commandline::

    winimportlib-exports G4OK G4OK_API  

https://cmake.org/Wiki/BuildingWinDLL

CMake will define G4OK_EXPORTS on Windows when it
configures to build a shared library. If you are going to use
another build system on windows or create the visual studio
projects by hand you need to define G4OK_EXPORTS when
building a DLL on windows.

*/

// TODO: probably mingw32 will need handling 

#if defined (_WIN32) 

   #if defined(G4OK_EXPORTS)
       #define  G4OK_API __declspec(dllexport)
   #else
       #define  G4OK_API __declspec(dllimport)
   #endif

#else

   #define G4OK_API  __attribute__ ((visibility ("default")))

#endif


