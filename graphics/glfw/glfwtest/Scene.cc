#include <GL/glew.h>
#include "Scene.hh"

#include "Shader.hh"
#include "Array.hh"
#include "Buffer.hh"
#include "IGeometry.hh"

#include "stdio.h"
#include "stdlib.h"

Scene::Scene()
 
  :
  m_shader(NULL),
  m_shaderdir(NULL)
{
}

Scene::~Scene()
{
}

GLuint Scene::upload(GLenum target, GLenum usage, Buffer* buffer)
{
    GLuint id ; 
    glGenBuffers(1, &id);
    glBindBuffer(target, id);
    glBufferData(target, buffer->getNumBytes(), buffer->getPointer(), usage);
    return id ; 
}

void Scene::setShaderDir(const char* dir)
{
    m_shaderdir = strdup(dir);
}

char* Scene::getShaderDir()
{
    return m_shaderdir ? m_shaderdir : getenv("SHADER_DIR") ;
}


void Scene::dump(const char* msg)
{
    printf("%s\n", msg );
    printf("nelem     %d \n", m_nelem);
    printf("vertices  %u \n", m_vertices);
    printf("colors    %u \n", m_colors);
    printf("indices   %u \n", m_indices);
    printf("shaderdir %s \n", getShaderDir());

    m_shader->dump(msg);
}

void Scene::init(IGeometry* geometry)
{
    glGenVertexArrays (1, &m_vao); // OSX: undefined without glew 
    glBindVertexArray (m_vao);     

    m_nelem    = geometry->getNumElements();
    m_vertices = upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW,  geometry->getVertices());
    m_colors   = upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW,  geometry->getColors());
    m_indices  = upload(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, geometry->getIndices());

    GLboolean normalized = GL_FALSE ; 
    GLsizei stride = 0 ;
    const GLvoid* offset = NULL ;
 
    // as there are two GL_ARRAY_BUFFER for vertices and colors need
    // to bind them again (despite bound in upload) in order to 
    // make the desired one active when create the VertexAttribPointer :
    // the currently active buffer being recorded "into" the VertexAttribPointer 

    glBindBuffer (GL_ARRAY_BUFFER, m_vertices);
    glVertexAttribPointer(vPosition, m_nelem, GL_FLOAT, normalized, stride, offset);
    glEnableVertexAttribArray (vPosition);   

    glBindBuffer (GL_ARRAY_BUFFER, m_colors);
    glVertexAttribPointer(vColor, m_nelem, GL_FLOAT, normalized, stride, offset);
    glEnableVertexAttribArray (vColor);   

    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, m_indices);

    // without 
    //     glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, m_indices);
    // got a blank despite being bound in the upload 
    // when VAO creation was after upload. It appears necessary to 
    // moving VAO creation to before the upload in order for it 
    // to capture this state.
    //
    // As there is only one GL_ELEMENT_ARRAY_BUFFER there is 
    // no need to repeat the bind, but doing so for clarity
    //

    m_shader = new Shader(getShaderDir());
    m_program = m_shader->getId();
    glUseProgram (m_program);       

    dump("Scene::init");
}

void Scene::draw()
{ 
    glBindVertexArray (m_vao);
    glDrawElements( GL_TRIANGLES, m_nelem, GL_UNSIGNED_INT, NULL ) ;
}


