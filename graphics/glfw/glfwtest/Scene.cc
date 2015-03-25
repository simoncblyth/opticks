#include <GL/glew.h>
#include "Scene.hh"
#include "Shader.hh"
#include "Array.hh"
#include "stdio.h"
#include "stdlib.h"

const float Scene::pvertex[] = {
   0.0f,  0.5f,  0.0f,
   0.5f, -0.5f,  0.0f,
  -0.5f, -0.5f,  0.0f
};

const float Scene::pcolor[] = {
  1.0f, 0.0f,  0.0f,
  0.0f, 1.0f,  0.0f,
  0.0f, 0.0f,  1.0f
};

const unsigned int Scene::pindex[] = {
      0,  1,  2
};


Scene::Scene()
{
    init();
}

Scene::~Scene()
{
}

void Scene::init()
{
    Array<float>* vertices = new Array<float>(9, &pvertex[0]);
    Array<float>* colors   = new Array<float>(9, &pcolor[0]);
    Array<unsigned int>*   indices  = new Array<unsigned int>(3,  &pindex[0]);

    GLint nelem = indices->getLength();

    vertices->upload(GL_ARRAY_BUFFER);
    colors->upload(GL_ARRAY_BUFFER);
    indices->upload(GL_ELEMENT_ARRAY_BUFFER);

    printf("vertices %u \n", vertices->getId());
    printf("colors   %u \n", colors->getId());
    printf("indices  %u \n", indices->getId());

    glGenVertexArrays (1, &m_vao); // OSX: undefined without glew 
    glBindVertexArray (m_vao);     

    GLboolean normalized = GL_FALSE ; 
    GLsizei stride = 0 ;
    const GLvoid* offset = NULL ;
 
    glEnableVertexAttribArray (0);   
    glBindBuffer (GL_ARRAY_BUFFER, vertices->getId());
    glVertexAttribPointer(0, nelem, GL_FLOAT, normalized, stride, offset);

    glEnableVertexAttribArray (1);   
    glBindBuffer (GL_ARRAY_BUFFER, colors->getId());
    glVertexAttribPointer(1, nelem, GL_FLOAT, normalized, stride, offset);

    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, indices->getId());

    char* glsldir = getenv("SHADER_DIR");
    m_shader = new Shader(glsldir);
    m_program = m_shader->getId();
    m_shader->dump();
}

void Scene::draw()
{ 
    glUseProgram (m_program);       
    glBindVertexArray (m_vao);
    glDrawElements( GL_TRIANGLES, 3, GL_UNSIGNED_INT, NULL ) ;
}


