#pragma once

#include <glm/vec4.hpp> 
#include <glm/vec2.hpp> 
#include <glm/mat4x4.hpp> 


#if defined (_WIN32) 

   #if defined(OGLRap_EXPORTS)
       #define  USEGLM_API __declspec(dllexport)
   #else
       #define  USEGLM_API __declspec(dllimport)
   #endif

#else

   #define USEGLM_API  __attribute__ ((visibility ("default")))

#endif


struct USEGLM_API UseGLM
{
    static glm::mat4 camera(float Translate, glm::vec2 const& Rotate) ;
};


 



