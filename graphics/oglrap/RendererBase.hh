#pragma once


class Composition ;
class Shader ; 


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
      void use_shader();   

  protected:
      Shader* m_shader ;
      GLuint m_program ;
      Composition* m_composition ;

  private:
      char* m_shaderdir ; 
      char* m_shadertag ; 

};


