#include <GL/glew.h>

#include "Rdr.hh"
#include "Prog.hh"
#include "Composition.hh"

// npy-
#include "NPY.hpp"
#include "VecNPY.hpp"
#include "MultiVecNPY.hpp"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


const char* Rdr::PRINT = "print" ; 



void Rdr::upload(MultiVecNPY* mvn)
{
    LOG(debug) << "Rdr::upload for shader tag " << getShaderTag() ;

    assert(mvn);

    //mvn->Print("Rdr::upload");    


    make_shader();  // need to compile and link shader for access to attribute locations

    glUseProgram(m_program);

    check_uniforms();


    NPY *npy(NULL);
    unsigned int count(0);

    for(unsigned int i=0 ; i<mvn->getNumVecs() ; i++)
    {
         VecNPY* vnpy = (*mvn)[i] ;

         // MultiVecNPY are constrained to all refer to the same underlying NPY 
         // so only do upload and m_buffer creation for the first 
 
         if(npy == NULL)
         {
             npy = vnpy->getNPY(); 
             count = vnpy->getCount();
             setCountDefault(count);

             upload(npy->getBytes(), npy->getNumBytes(0));

             npy->setBufferId(m_buffer); // record OpenGL buffer id with the data for convenience

         }
         else 
         {
             assert(npy == vnpy->getNPY());     // make sure all match
             LOG(debug) << "Rdr::upload counts, prior: " << count << " current: " << vnpy->getCount() ; 
             assert(count == vnpy->getCount());
         }

         address(vnpy); 
    }
}

void Rdr::upload(void* data, unsigned int nbytes)
{
    glGenVertexArrays (1, &m_vao); 
    glBindVertexArray (m_vao);     

    glGenBuffers(1, &m_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, m_buffer);
    glBufferData(GL_ARRAY_BUFFER, nbytes, data, GL_STATIC_DRAW );
}



void Rdr::download( NPY* npy )
{
    int buffer_id = npy ? npy->getBufferId() : -1 ;
    if(buffer_id == -1) return ;

    LOG(info)<< "Rdr::download " << " buffer_id " << buffer_id  ;

    GLenum target = GL_ARRAY_BUFFER ;
    GLenum access = GL_READ_ONLY ; 

    glBindBuffer( target, buffer_id );

    void* ptr = glMapBuffer( target, access );  
    npy->read(ptr);
    glUnmapBuffer(target);

    glBindBuffer(target, 0 );

}


void Rdr::address(VecNPY* vnpy)
{
    const char* name = vnpy->getName();  
    GLint location = m_shader->attribute(name, false);
    if(location == -1)
    {
         LOG(warning)<<"Rdr::address failed to find active attribute for VecNPY named " << name 
                     << " in shader " << getShaderTag() ;
         return ;
    }

    GLenum type ;              //  of each component in the array
    switch(vnpy->getType())
    {
        case 'f':type = GL_FLOAT        ; break ;
        case 'i':type = GL_INT          ; break ;
        case 'u':type = GL_UNSIGNED_INT ; break ;
        default: assert(0)              ; break ; 
    }

    GLuint       index = location  ;       //  generic vertex attribute to be modified
    GLint         size = vnpy->getSize() ; //  number of components per generic vertex attribute, must be 1,2,3,4
    GLboolean     norm = GL_FALSE ; 
    GLsizei       stride = vnpy->getStride();  ;         // byte offset between consecutive generic vertex attributes, or 0 for tightly packed
    const GLvoid* offset = (const GLvoid*)vnpy->getOffset() ;      

    // offset of the first component of the first generic vertex attribute 
    // in the array in the data store of the buffer currently bound to GL_ARRAY_BUFFER target

    if( type == GL_INT || type == GL_UNSIGNED_INT  )
    {
        glVertexAttribIPointer(index, size, type, stride, offset);
    }
    else if( type == GL_FLOAT )
    {
        glVertexAttribPointer(index, size, type, norm, stride, offset);
    }
    else
    {
        assert(0);
    }
    glEnableVertexAttribArray(index);

}


void Rdr::check_uniforms()
{
    bool required = false ; 
    m_mvp_location = m_shader->uniform("ModelViewProjection", required) ; 
    m_mv_location = m_shader->uniform("ModelView", required );     
    m_selection_location = m_shader->uniform("Selection", required );     
    m_flags_location = m_shader->uniform("Flags", required );     
    m_param_location = m_shader->uniform("Param", required );     

    // the "tag" argument of the Rdr identifies the GLSL code being used
    // determining which uniforms are required 

    LOG(info) << "Rdr::check_uniforms "
              << " mvp " << m_mvp_location
              << " mv " << m_mv_location 
              << " sel " << m_selection_location 
              << " flg " << m_flags_location 
              << " param " << m_param_location 
              ;

}


void Rdr::update_uniforms()
{
    if(m_composition)
    {
        m_composition->update() ;
        glUniformMatrix4fv(m_mv_location, 1, GL_FALSE,  m_composition->getWorld2EyePtr());
        glUniformMatrix4fv(m_mvp_location, 1, GL_FALSE, m_composition->getWorld2ClipPtr());

        glm::ivec4 sel = m_composition->getSelection();
        glUniform4i(m_selection_location, sel.x, sel.y, sel.z, sel.w  );    

        glm::ivec4 flg = m_composition->getFlags();
        glUniform4i(m_flags_location, flg.x, flg.y, flg.z, flg.w  );    

        glm::vec4 par = m_composition->getParam();
        glUniform4f(m_param_location, par.x, par.y, par.z, par.w  );    




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

    glDrawArrays( GL_POINTS, first_, count_ );
    //glDrawArrays( GL_LINES, first_, count_ );

    glBindVertexArray(0);

    glUseProgram(0);
}





