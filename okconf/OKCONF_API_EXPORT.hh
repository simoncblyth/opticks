
#pragma once

#if defined (_WIN32) 

   #if defined(OKCONF_EXPORTS)
       #define  OKCONF_API __declspec(dllexport)
   #else
       #define  OKCONF_API __declspec(dllimport)
   #endif

#else

   #define OKCONF_API  __attribute__ ((visibility ("default")))

#endif


