
#pragma once

#if defined (_WIN32) 

   #if defined(UseInstance_EXPORTS)
       #define  USEINSTANCE_API __declspec(dllexport)
   #else
       #define  USEINSTANCE_API __declspec(dllimport)
   #endif

#else

   #define USEINSTANCE_API  __attribute__ ((visibility ("default")))

#endif


