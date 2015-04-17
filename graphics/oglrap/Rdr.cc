#include <GL/glew.h>

#include "Rdr.hh"
#include "Prog.hh"
#include "Composition.hh"

// npy-
#include "NPY.hpp"
#include "VecNPY.hpp"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


const char* Rdr::PRINT = "print" ; 

Rdr::Rdr(const char* tag)
    :
    RendererBase(tag),
    m_vao(0),
    m_buffer(0),
    m_countdefault(0),
    m_composition(NULL),
    m_mv_location(-1),
    m_mvp_location(-1)
{
}

void Rdr::upload(VecNPY* vnpy, bool debug)
{
    if(debug) vnpy->dump("Rdr::upload");
    upload( vnpy->getBytes(), vnpy->getNumBytes(), vnpy->getStride(), vnpy->getOffset(), vnpy->getCount() );
}

void Rdr::upload(NPY* npy, unsigned int j, unsigned int k )
{
    void* bytes = npy->getBytes();
    unsigned int nbytes = npy->getNumBytes(0);      // from dimension 0, ie total bytes
    unsigned int stride = npy->getNumBytes(1);      // from dimension 1, ie item bytes  
    unsigned int offset = npy->getByteIndex(0,j,k); // length of 3rd dimension is usually 4 for efficient float4/quad handling  
    unsigned int count  = npy->getShape(0); 

    upload( bytes, nbytes, stride, offset, count );
}

void Rdr::upload(void* data, unsigned int nbytes, unsigned int stride, unsigned long offset, unsigned int countdefault)
{
    setCountDefault(countdefault);

    glGenVertexArrays (1, &m_vao); 
    glBindVertexArray (m_vao);     

    glGenBuffers(1, &m_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, m_buffer);
    glBufferData(GL_ARRAY_BUFFER, nbytes, data, GL_STATIC_DRAW );


    LOG(info) << "Rdr::upload " 
              << " m_vao " << m_vao 
              << " m_buffer " << m_buffer 
              << " nbytes " << nbytes 
              << " stride " << stride 
              << " offset " << offset ; 


    GLuint index = vRdrPosition ;       //  generic vertex attribute to be modified
    GLint  size = 3          ;          //  number of components per generic vertex attribute, must be 1,2,3,4
    GLenum type = GL_FLOAT   ;          //  of each component in the array
    GLboolean normalized = GL_FALSE ; 
    GLsizei stride_ = stride ;          // byte offset between consecutive generic vertex attributes, or 0 for tightly packed
    const GLvoid* offset_ = (const GLvoid*)offset ;      

    // offset of the first component of the first generic vertex attribute 
    // in the array in the data store of the buffer currently bound to GL_ARRAY_BUFFER target

    glVertexAttribPointer(index, size, type, normalized, stride_, offset_);
    glEnableVertexAttribArray(index);

    make_shader();

    // the "tag" argument of the Rdr identifies the GLSL code being used
    // determining which uniforms are required 

    m_mvp_location = m_shader->uniform("ModelViewProjection", false) ; 
    m_mv_location = m_shader->uniform("ModelView", false);      // not required

    LOG(info) << "Rdr::make_shader "
              << " mvp " << m_mvp_location
              << " mv " << m_mv_location ;


    glUseProgram(m_program);
}



void Rdr::update_uniforms()
{
    if(m_composition)
    {
        m_composition->update() ;
        glUniformMatrix4fv(m_mv_location, 1, GL_FALSE,  m_composition->getWorld2EyePtr());
        glUniformMatrix4fv(m_mvp_location, 1, GL_FALSE, m_composition->getWorld2ClipPtr());
    } 
    else
    { 
        glm::mat4 identity ; 
        glUniformMatrix4fv(m_mv_location, 1, GL_FALSE, glm::value_ptr(identity));
        glUniformMatrix4fv(m_mvp_location, 1, GL_FALSE, glm::value_ptr(identity));
    }
}


void Rdr::render(unsigned int count, unsigned int first)
{
    glUseProgram(m_program);

    update_uniforms();

    glBindVertexArray(m_vao);

    GLint   first_ = first  ;                            // starting index in the enabled arrays
    GLsizei count_ = count ? count : m_countdefault  ;   // number of indices to be rendered

    //glDrawArrays( GL_POINTS, first_, count_ );
    glDrawArrays( GL_LINES, first_, count_ );

    glBindVertexArray(0);

    glUseProgram(0);
}


