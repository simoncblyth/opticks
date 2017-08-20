#pragma once
#include <cstddef>

class Prog ;

#include "OGLRAP_API_EXPORT.hh"
class OGLRAP_API RendererBase {
   public:
      RendererBase(const char* tag, const char* dir=NULL, const char* incl_path=NULL);
      const char* getShaderTag() const ; 
      const char* getShaderDir() const ; 
      const char* getInclPath() const ; 

  public:
      void make_shader();   
  protected:
      Prog*   m_shader ;
      int     m_program ;
  private:
      const char* m_shaderdir ; 
      const char* m_shadertag ; 
      const char* m_incl_path ; 

};

