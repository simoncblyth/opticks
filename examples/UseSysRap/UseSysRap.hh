#pragma once

#if defined (_WIN32) 

   #if defined(EXPORTS)
       #define  API __declspec(dllexport)
   #else
       #define  API __declspec(dllimport)
   #endif

#else

   #define API  __attribute__ ((visibility ("default")))

#endif



extern API void UseSysRap(const char* msg); 

