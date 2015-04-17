#pragma once

class Composition ;
class Prog ;


class RendererBase {
   public:
      RendererBase(const char* tag);

      void setComposition(Composition* composition);
      void setShaderDir(const char* dir);
      void setShaderTag(const char* tag);

      Composition* getComposition(); 
      char* getShaderDir(); 
      char* getShaderTag(); 

  public:
      void make_shader();   
      void update_uniforms();   
      void dump(void* data, unsigned int nbytes, unsigned int stride, unsigned long offset, unsigned int count );

  protected:
      Prog*   m_shader ;
      GLuint m_program ;
      Composition* m_composition ;

  private:
      char* m_shaderdir ; 
      char* m_shadertag ; 
      GLint  m_mv_location ;
      GLint  m_mvp_location ;


};


