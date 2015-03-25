#ifndef GEOMETRY_H
#define GEOMETRY_H

// http://antongerdelan.net/opengl/hellotriangle.html
class Shader ; 
class VertexBuffer ;

class Geometry {
  public:
      static const float points[] ;

      Geometry();
      virtual ~Geometry();

      void initVBO(unsigned int length, const float* values);
      void initVAO(); // VAO collects details of all the VBO
      void initShader();
      void draw();

  private:
      GLuint m_vao ; 
      VertexBuffer* m_vbo ; 
      GLuint m_vs ;
      GLuint m_fs ;
      Shader* m_shader ;
      GLuint m_program ;
};      


#endif



 
