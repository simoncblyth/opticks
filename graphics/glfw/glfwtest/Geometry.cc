#include <GL/glew.h>
#include "Geometry.hh"
#include "Shader.hh"
#include "Array.hh"
#include "VertexBuffer.hh"
#include "VertexAttribute.hh"

const float Geometry::points[] = {
   0.0f,  0.5f,  0.0f,
   0.5f, -0.5f,  0.0f,
  -0.5f, -0.5f,  0.0f
};


Geometry::Geometry()
{
    Array* vertices = new Array(9, &points[0]);
    m_vbo = new VertexBuffer( vertices, NULL );

    initVAO();
    initShader();
}
Geometry::~Geometry()
{
}


void Geometry::initVAO() // VAO collects details of all the VBO
{
    glGenVertexArrays (1, &m_vao); // OSX: undefined without glew 
    glBindVertexArray (m_vao);     // OSX: undefined without glew 

    unsigned int slot = 0 ;
    glBindBuffer (GL_ARRAY_BUFFER, m_vbo->getHandle());
    VertexAttribute* attrib = new VertexAttribute(slot, 3, GL_FLOAT, GL_FALSE, 0, NULL);
    attrib->enable();
}


void Geometry::initShader()
{
    m_shader = new Shader();
    m_program = m_shader->getProgram();
    m_shader->dump();
} 


void Geometry::draw()
{ 
    glUseProgram (m_program);       
    glBindVertexArray (m_vao); // OSX:undefined without glew 
    glDrawArrays (GL_TRIANGLES, 0, 3);
}


