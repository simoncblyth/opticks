#pragma once
#include <cstddef>

class Prog ;

#include "OGLRAP_API_EXPORT.hh"
class OGLRAP_API RendererBase {
   public:
      RendererBase(const char* tag, const char* dir=NULL, const char* incl_path=NULL);
      char* getShaderDir(); 
      char* getShaderTag(); 
  public:
      void make_shader();   
  protected:
      Prog*   m_shader ;
      //GLuint  m_program ;
      int     m_program ;
  private:
      char* m_shaderdir ; 
      char* m_shadertag ; 
      char* m_incl_path ; 

};

