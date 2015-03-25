#ifndef SHADER_H
#define SHADER_H

// http://antongerdelan.net/opengl/shaders.html

class Shader {
  public:
      static const char* vertex_shader;
      static const char* fragment_shader;

      Shader();
      virtual ~Shader();
      GLuint getProgram(); 
      void dump();
      bool isValid();

   private:
      void init();
      void compile(GLuint index);
      void link(GLuint index);

  private:
      GLuint m_vs ;
      GLuint m_fs ;
      GLuint m_program ;
};      


#endif

