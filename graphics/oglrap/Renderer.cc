#include <GL/glew.h>

#include "Renderer.hh"
#include "Shader.hh"
#include "Composition.hh"

// npy-
#include "GLMPrint.hpp"


// ggeo
#include "GArray.hh"
#include "GBuffer.hh"
#include "GDrawable.hh"

#include "stdio.h"
#include "stdlib.h"

const char* Renderer::PRINT = "print" ; 

Renderer::Renderer(const char* tag)
    :
    RendererBase(tag),
    m_drawable(NULL),
    m_draw_count(0),
    m_texcoords(0),
    m_has_tex(false) 
{
}

Renderer::~Renderer()
{
}

void Renderer::configureI(const char* name, std::vector<int> values )
{
    if(values.empty()) return ; 
    if(strcmp(name, PRINT)==0) Print("Renderer::configureI");
}


void Renderer::setDrawable(GDrawable* drawable, bool debug)
{
    m_drawable = drawable ;
    gl_upload_buffers(debug);
}


void Renderer::gl_upload_buffers(bool debug)
{
    // as there are two GL_ARRAY_BUFFER for vertices and colors need
    // to bind them again (despite bound in upload) in order to 
    // make the desired one active when create the VertexAttribPointer :
    // the currently active buffer being recorded "into" the VertexAttribPointer 
    //
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
    assert(m_drawable);

    glGenVertexArrays (1, &m_vao); // OSX: undefined without glew 
    glBindVertexArray (m_vao);     

    GBuffer* vbuf = m_drawable->getVerticesBuffer();
    GBuffer* nbuf = m_drawable->getNormalsBuffer();
    GBuffer* cbuf = m_drawable->getColorsBuffer();
    GBuffer* ibuf = m_drawable->getIndicesBuffer();
    GBuffer* tbuf = m_drawable->getTexcoordsBuffer();
    setHasTex(tbuf != NULL);

    if(debug)
    {
        RendererBase::dump( vbuf->getPointer(),vbuf->getNumBytes(),vbuf->getNumElements()*sizeof(float),0,vbuf->getNumItems() ); 
    }

    assert(vbuf->getNumBytes() == cbuf->getNumBytes());
    assert(nbuf->getNumBytes() == cbuf->getNumBytes());

    m_vertices  = upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW,  vbuf );
    m_normals   = upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW,  nbuf );
    m_colors    = upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW,  cbuf );
    if(hasTex())
    {
        m_texcoords = upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW,  tbuf );
    }

    m_indices  = upload(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, ibuf );
    m_indices_count = ibuf->getNumItems(); // number of indices

    GLboolean normalized = GL_FALSE ; 
    GLsizei stride = 0 ;
    const GLvoid* offset = NULL ;
 
    // the vbuf and cbuf NumElements refer to the number of elements 
    // within the vertex and color items ie 3 in both cases

    glBindBuffer (GL_ARRAY_BUFFER, m_vertices);
    glVertexAttribPointer(vPosition, vbuf->getNumElements(), GL_FLOAT, normalized, stride, offset);
    glEnableVertexAttribArray (vPosition);  

    glBindBuffer (GL_ARRAY_BUFFER, m_normals);
    glVertexAttribPointer(vNormal, nbuf->getNumElements(), GL_FLOAT, normalized, stride, offset);
    glEnableVertexAttribArray (vNormal);  

    glBindBuffer (GL_ARRAY_BUFFER, m_colors);
    glVertexAttribPointer(vColor, cbuf->getNumElements(), GL_FLOAT, normalized, stride, offset);
    glEnableVertexAttribArray (vColor);   

    if(hasTex())
    {
        glBindBuffer (GL_ARRAY_BUFFER, m_texcoords);
        glVertexAttribPointer(vTexcoord, tbuf->getNumElements(), GL_FLOAT, normalized, stride, offset);
        glEnableVertexAttribArray (vTexcoord);   
    }

    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, m_indices);

    
    make_shader();

    if(hasTex())
    {
        m_sampler_location = m_shader->getSamplerLocation();
    }


    glUseProgram(m_program);

}


GLuint Renderer::upload(GLenum target, GLenum usage, GBuffer* buffer)
{
    //buffer->Summary("Renderer::upload");
    GLuint id ; 
    glGenBuffers(1, &id);
    glBindBuffer(target, id);
    glBufferData(target, buffer->getNumBytes(), buffer->getPointer(), usage);
    return id ; 
}


void Renderer::render()
{ 
    glUseProgram(m_program);

    update_uniforms();

    glBindVertexArray (m_vao);

    glDrawElements( GL_TRIANGLES, m_indices_count, GL_UNSIGNED_INT, NULL ) ; // indices_count would be 3 for a single triangle 

    m_draw_count += 1 ; 

    glUseProgram(0);
}


void Renderer::dump(const char* msg)
{
    printf("%s\n", msg );
    printf("vertices  %u \n", m_vertices);
    printf("normals   %u \n", m_normals);
    printf("colors    %u \n", m_colors);
    printf("indices   %u \n", m_indices);
    printf("nelem     %d \n", m_indices_count);
    printf("hasTex    %d \n", hasTex());
    printf("shaderdir %s \n", getShaderDir());
    printf("shadertag %s \n", getShaderTag());

    m_shader->dump(msg);
}

void Renderer::Print(const char* msg)
{
    printf("Renderer::%s tag %s nelem %d vao %d \n", msg, getShaderTag(), m_indices_count, m_vao );
}


