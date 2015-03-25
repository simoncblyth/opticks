#ifndef SHADER_H
#define SHADER_H

#include <string>
// http://antongerdelan.net/opengl/shaders.html

class Shader {
  public:
      static const char* vertex_shader;
      static const char* fragment_shader;

      Shader(const char* dir, const char* vname="vert.glsl", const char* fname="frag.glsl");
      virtual ~Shader();
      GLuint getId(); 
      void dump();
      bool isValid();

   private:
      void init(const std::string& vsrc, const std::string& fsrc);
      void compile(GLuint index);
      void link(GLuint index);

  private:
      GLuint m_vs ;
      GLuint m_fs ;
      GLuint m_program ;
};      


#endif

