#pragma once
#include <cstddef>

class Prog ;

/*
RendererBase
==============

Hmm ShaderBase would be a better name, in light of transform feedback 
(and maybe compute shaders in future).


*/


#include "OGLRAP_API_EXPORT.hh"
class OGLRAP_API RendererBase {
   public:
      RendererBase(const char* tag, const char* dir=NULL, const char* incl_path=NULL, bool ubo=false);
      const char* getShaderTag() const ; 
      const char* getShaderDir() const ; 
      const char* getInclPath() const ; 

      void setVerbosity(unsigned verbosity);
      void setNoFrag(bool nofrag);
  public:
      void make_shader();   
  public: 
      // split the make, for transform feedback where need to setup varyings between create and link 
      void create_shader();   
      void link_shader();   
  protected:
      Prog*     m_shader ;
      int       m_program ;
      unsigned  m_verbosity ; 
   private:
      const char* m_shaderdir ; 
      const char* m_shadertag ; 
      const char* m_incl_path ; 

};

