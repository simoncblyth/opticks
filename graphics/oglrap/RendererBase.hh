#pragma once


class Composition ;
class Shader ; 
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
      Shader* m_shader ;
      GLuint m_program ;
      Composition* m_composition ;

  protected:
      // testing new shader handling 
      Prog* m_shaderprog ;

  private:
      char* m_shaderdir ; 
      char* m_shadertag ; 
      GLint  m_mv_location ;
      GLint  m_mvp_location ;


};


