#include <GL/glew.h>

#include "Rdr.hh"
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
    m_buffer(0) 
{
}

void Rdr::configureI(const char* name, std::vector<int> values )
{
    if(values.empty()) return ; 
    if(strcmp(name, PRINT)==0) Print("Rdr::configureI");
}

void Rdr::Print(const char* msg)
{
    printf("%s\n", msg);
}


void Rdr::upload(void* data, unsigned int nbytes, unsigned int stride, unsigned long offset )
{

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
    GLint  size = 3          ;       //  number of components per generic vertex attribute, must be 1,2,3,4
    GLenum type = GL_FLOAT   ;       //  of each component in the array
    GLboolean normalized = GL_FALSE ; 
    GLsizei stride_ = stride ;                // byte offset between consecutive generic vertex attributes, or 0 for tightly packed
    const GLvoid* offset_ = (const GLvoid*)offset ;      

    //  offset of the first component of the first generic vertex attribute 
    // in the array in the data store of the buffer currently bound to GL_ARRAY_BUFFER target

    glVertexAttribPointer(index, size, type, normalized, stride_, offset_);
    glEnableVertexAttribArray(index);

    make_shader();

    glUseProgram(m_program);
}




void Rdr::render(unsigned int count, unsigned int first)
{
    glUseProgram(m_program);

    update_uniforms();

    glBindVertexArray(m_vao);

    GLint   first_ = first  ;  // starting index in the enabled arrays
    GLsizei count_ = count ;   // number of indices to be rendered

    //glDrawArrays( GL_POINTS, first_, count_ );
    glDrawArrays( GL_LINES, first_, count_ );

    glBindVertexArray(0);

    glUseProgram(0);
}



