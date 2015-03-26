#include <GL/glew.h>
#include "VertexAttribute.hh"

VertexAttribute::VertexAttribute(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid * offset)
  : 
  m_index(index),
  m_size(size),
  m_type(type),
  m_normalized(normalized),
  m_stride(stride),
  m_offset(offset)
{
  // buffer vital stats
}

VertexAttribute::~VertexAttribute()
{
}


void VertexAttribute::enable()
{
    glVertexAttribPointer ( m_index, m_size, m_type, m_normalized, m_stride, m_offset );
    glEnableVertexAttribArray (m_index);   
}



