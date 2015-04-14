#include <GL/glew.h>

#include "Renderer.hh"
#include "Shader.hh"
#include "Composition.hh"
#include "Common.hh"


// ggeo
#include "GArray.hh"
#include "GBuffer.hh"
#include "GDrawable.hh"

#include "stdio.h"
#include "stdlib.h"

#include <glm/glm.hpp>  
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/type_ptr.hpp>


const char* Renderer::PRINT = "print" ; 

Renderer::Renderer(const char* tag)
    :
    m_shader(NULL),
    m_shaderdir(NULL),
    m_shadertag(NULL),
    m_drawable(NULL),
    m_composition(NULL),
    m_draw_count(0),
    m_texcoords(0),
    m_has_tex(false) 
{
    setShaderTag(tag);
}

Renderer::~Renderer()
{
}

void Renderer::configureI(const char* name, std::vector<int> values )
{
    if(values.empty()) return ; 
    if(strcmp(name, PRINT)==0) Print("Renderer::configureI");
}


void Renderer::setDrawable(GDrawable* drawable)
{
    m_drawable = drawable ;
    gl_upload_buffers();
}
void Renderer::setComposition(Composition* composition)
{
    m_composition = composition ;
}
void Renderer::setShaderDir(const char* dir)
{
    m_shaderdir = strdup(dir);
}
void Renderer::setShaderTag(const char* tag)
{
    m_shadertag = strdup(tag);
}


Composition* Renderer::getComposition()
{
    return m_composition ;
}
char* Renderer::getShaderDir()
{
    return m_shaderdir ? m_shaderdir : getenv("SHADER_DIR") ;
}
char* Renderer::getShaderTag()
{
    return m_shadertag ? m_shadertag : getenv("SHADER_TAG") ;
}





void Renderer::gl_upload_buffers()
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

    if(m_composition)
    { 
        float* model_to_world  = (float*)m_drawable->getModelToWorldBuffer()->getPointer();
        float extent = m_drawable->getExtent();
        m_composition->setModelToWorld_Extent(model_to_world, extent);
    }

    glGenVertexArrays (1, &m_vao); // OSX: undefined without glew 
    glBindVertexArray (m_vao);     

    GBuffer* vbuf = m_drawable->getVerticesBuffer();
    GBuffer* nbuf = m_drawable->getNormalsBuffer();
    GBuffer* cbuf = m_drawable->getColorsBuffer();
    GBuffer* ibuf = m_drawable->getIndicesBuffer();
    GBuffer* tbuf = m_drawable->getTexcoordsBuffer();
    setHasTex(tbuf != NULL);

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

    m_shader = new Shader(getShaderDir(), getShaderTag());
    m_program = m_shader->getId();
    m_mvp_location = m_shader->getMVPLocation();
    m_mv_location = m_shader->getMVLocation();

    if(hasTex())
    {
        m_sampler_location = m_shader->getSamplerLocation();
    }

    m_shader->use();

    //dump("Renderer::init");
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

    if(m_composition)
    {
        m_composition->update() ;
        // could cache the ptrs they aint changing 
        glUniformMatrix4fv(m_mv_location, 1, GL_FALSE, glm::value_ptr(m_composition->getWorld2Eye()));
        glUniformMatrix4fv(m_mvp_location, 1, GL_FALSE, glm::value_ptr(m_composition->getWorld2Clip()));
    } 
    else
    { 
        glm::mat4 identity ; 
        glUniformMatrix4fv(m_mv_location, 1, GL_FALSE, glm::value_ptr(identity));
        glUniformMatrix4fv(m_mvp_location, 1, GL_FALSE, glm::value_ptr(identity));
    }


    m_shader->use();

    glBindVertexArray (m_vao);

    //GLsizei count = m_indices_count ; 

    glDrawElements( GL_TRIANGLES, m_indices_count, GL_UNSIGNED_INT, NULL ) ;

    m_draw_count += 1 ; 

    // count is the number of indices (which point at vertices) to form into triangle faces
    // in the single triangle Demo this would be 3 
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


