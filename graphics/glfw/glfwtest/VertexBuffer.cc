#include <GL/glew.h>
#include "VertexBuffer.hh"
#include "Array.hh"

VertexBuffer::VertexBuffer( Array* vertices, Array* indices )
  : 
  m_vertices(vertices),
  m_indices(indices)
{
    glGenBuffers(1, &m_handle);
    glBindBuffer(GL_ARRAY_BUFFER, m_handle);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*vertices->getLength(), vertices->getValues(), GL_STATIC_DRAW);
}

VertexBuffer::~VertexBuffer()
{
}

GLuint VertexBuffer::getHandle()
{
  return m_handle ;  
}

 
