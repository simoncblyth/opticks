#pragma once

#include <string>
#include "OGLRAP_API_EXPORT.hh"

struct OGLRAP_API G
{
     static const char* GL_INVALID_ENUM_ ; 
     static const char* GL_INVALID_VALUE_ ; 
     static const char* GL_INVALID_OPERATION_ ; 
     static const char* GL_STACK_OVERFLOW_ ; 
     static const char* GL_STACK_UNDERFLOW_ ; 
     static const char* GL_OUT_OF_MEMORY_ ; 
     static const char* GL_INVALID_FRAMEBUFFER_OPERATION_ ; 
     static const char* GL_CONTEXT_LOST_ ;
     static const char* OTHER_ ;



     static const char* Err( GLenum err );
     static bool ErrCheck(const char* msg, bool harikari ) ;


     static const char* GL_VERTEX_SHADER_ ; 
     static const char* GL_GEOMETRY_SHADER_ ; 
     static const char* GL_FRAGMENT_SHADER_ ; 

     static const char* Shader( GLenum type );



};
 
