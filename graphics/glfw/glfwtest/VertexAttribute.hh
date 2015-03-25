#ifndef VERTEXATTRIBUTE_H
#define VERTEXATTRIBUTE_H

// little need for this with OpenGL 4
// as VAOs now remember their attributes  

class VertexAttribute {
   public:
      VertexAttribute(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid * offset);
      virtual ~VertexAttribute();
      void enable();

  private:
      GLuint m_index ;
      GLint  m_size ;
      GLenum m_type ;
      GLboolean m_normalized ; 
      GLsizei m_stride ; 
      const GLvoid* m_offset ;


};

#endif
