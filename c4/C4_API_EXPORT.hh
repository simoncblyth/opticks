#pragma once

#if defined (_WIN32) 

   #if defined(C4_EXPORTS)
       #define  C4_API __declspec(dllexport)
   #else
       #define  C4_API __declspec(dllimport)
   #endif

#else

   #define C4_API  __attribute__ ((visibility ("default")))

#endif


