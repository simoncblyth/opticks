#pragma once

#if defined (_WIN32) 

   #if defined(SysRap_EXPORTS)
       #define  DEMO_API __declspec(dllexport)
   #else
       #define  DEMO_API __declspec(dllimport)
   #endif

#else

   #define DEMO_API  __attribute__ ((visibility ("default")))

#endif


