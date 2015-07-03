#include <GL/glew.h>

#include <iomanip>

#include "Device.hh"
#include "Rdr.hh"
#include "Prog.hh"
#include "Composition.hh"

// npy-
#include "NPY.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"
#include "stringutil.hpp"
#include "GLMPrint.hpp"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


const char* Rdr::PRINT = "print" ; 

void Rdr::setPrimitive(Primitive_t prim )
{
    switch( prim )
    {
        case        POINTS: m_primitive = GL_POINTS      ; break ; 
        case         LINES: m_primitive = GL_LINES       ; break ; 
        case    LINE_STRIP: m_primitive = GL_LINE_STRIP  ; break ; 
        default           : assert(0)                    ; break ;
    }
}


void Rdr::upload(MultiViewNPY* mvn)
{
    // MultiViewNPY are constrained to all refer to the same underlying NPY 
    // so only do upload and m_buffer creation for the first 

    assert(mvn);

    // need to compile and link shader for access to attribute locations
    if(m_first_upload)
    {
        make_shader();  
        glUseProgram(m_program);
        check_uniforms();
        log("Rdr::upload FIRST m_program:",m_program); 

    }    

    unsigned int count(0);
    NPYBase* npy(NULL);

    for(unsigned int i=0 ; i<mvn->getNumVecs() ; i++)
    {
        ViewNPY* vnpy = (*mvn)[i] ;
        if(npy == NULL)
        {
            count = vnpy->getCount();

            if(m_first_upload)
            {
                setCountDefault(count);
            }
            else
            {
                assert(count == getCountDefault() && "subsequent Rdr::uploads must have same count as first");
            }

            npy = vnpy->getNPY(); 
            upload(npy);      // duplicates are not re-uploaded
        }
        else
        {
            assert(npy == vnpy->getNPY());     
            LOG(debug) << "Rdr::upload counts, prior: " << count << " current: " << vnpy->getCount() ; 
            assert(count == vnpy->getCount());
        } 
        address(vnpy); 
    }


    if(m_first_upload)
    {
        m_first_upload = false ; 
    }
}



void Rdr::log(const char* msg, int value)
{
    LOG(info)
                 << "Rdr::log " 
                 << std::setw(10) << getShaderTag() 
                 << " "
                 << msg  
                 << value ;
 
}


void Rdr::prepare_vao()
{
    if(!m_vao_generated)
    {
        glGenVertexArrays (1, &m_vao); 
        m_vao_generated = true ; 
        log("prepare_vao : generate m_vao:", m_vao);
   }

    log("prepare_vao : bind m_vao:", m_vao);
    glBindVertexArray (m_vao);     

}


void Rdr::upload(NPYBase* npy)
{
    // handles case of multiple mvn referring to the same buffer without data duplication,
    // by maintaining a list of NPYBase which have been uploaded to the Device

    prepare_vao();

    void* aux = npy->getAux();
    if(aux)
    {
        assert(npy->getType() == NPYBase::UCHAR );
       // hmm how to avoid CUDA dependency for oglrap-
    }

    if(m_device->isUploaded(npy))
    {
        GLuint buffer_id = npy->getBufferId();
        log("Rdr::upload BindBuffer to preexisting buffer_id:",buffer_id)  ;
        assert(buffer_id > 0);
        glBindBuffer(GL_ARRAY_BUFFER, buffer_id);
    }
    else
    {
        void* data = npy->getBytes();
        unsigned int nbytes = npy->getNumBytes(0) ;

        GLuint buffer_id ;  
        glGenBuffers(1, &buffer_id);
        glBindBuffer(GL_ARRAY_BUFFER, buffer_id);

        if(aux)
        {
            log("Rdr::upload using GL_DYNAMIC_DRAW for aux enabled buffer ", buffer_id);
            glBufferData(GL_ARRAY_BUFFER, nbytes, data, GL_DYNAMIC_DRAW );
        }
        else
        {
            glBufferData(GL_ARRAY_BUFFER, nbytes, data, GL_STATIC_DRAW );
        }

        log("Rdr::upload BufferData gen buffer_id:", buffer_id ); 

        npy->setBufferId(buffer_id); 
        m_device->add(npy);
    }
}




/*
   multiple buffers

   http://stackoverflow.com/questions/14249634/opengl-vaos-and-multiple-buffers

   appropriate ordering is


        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, buffer1);
        glVertexAttribPointer(0, ...);
        glVertexAttribPointer(1, ...);

        glBindBuffer(GL_ARRAY_BUFFER, buffer2);
        glVertexAttribPointer(2, ...);



*/


void* Rdr::mapbuffer( int buffer_id, GLenum target )
{
    LOG(debug)<< "Rdr::mapbuffer " << " buffer_id " << buffer_id  ;
    if(buffer_id == -1) return NULL ;
    GLenum access = GL_READ_ONLY ; 
    glBindBuffer( target, buffer_id );
    void* ptr = glMapBuffer( target, access );  
    return ptr ;
}

void Rdr::unmapbuffer(GLenum target)
{
    glUnmapBuffer(target);
    glBindBuffer(target, 0 );
}


void Rdr::address(ViewNPY* vnpy)
{
    const char* name = vnpy->getName();  
    GLint location = m_shader->attribute(name, false);
    if(location == -1)
    {
         LOG(debug)<<"Rdr::address failed to find active attribute for ViewNPY named " << name 
                     << " in shader " << getShaderTag() ;
         return ;
    }

    GLenum type ;              //  of each component in the array
    switch(vnpy->getType())
    {
        case ViewNPY::BYTE:                         type = GL_BYTE           ; break ;
        case ViewNPY::UNSIGNED_BYTE:                type = GL_UNSIGNED_BYTE  ; break ;
        case ViewNPY::SHORT:                        type = GL_SHORT          ; break ;
        case ViewNPY::UNSIGNED_SHORT:               type = GL_UNSIGNED_SHORT ; break ;
        case ViewNPY::INT:                          type = GL_INT            ; break ;
        case ViewNPY::UNSIGNED_INT:                 type = GL_UNSIGNED_INT   ; break ;
        case ViewNPY::HALF_FLOAT:                   type = GL_HALF_FLOAT     ; break ;
        case ViewNPY::FLOAT:                        type = GL_FLOAT          ; break ;
        case ViewNPY::DOUBLE:                       type = GL_DOUBLE         ; break ;
        case ViewNPY::FIXED:                        type = GL_FIXED                        ; break ;
        case ViewNPY::INT_2_10_10_10_REV:           type = GL_INT_2_10_10_10_REV           ; break ; 
        case ViewNPY::UNSIGNED_INT_2_10_10_10_REV:  type = GL_UNSIGNED_INT_2_10_10_10_REV  ; break ; 
        //case ViewNPY::UNSIGNED_INT_10F_11F_11F_REV: type = GL_UNSIGNED_INT_10F_11F_11D_REV ; break ; 
        default: assert(0)                                                                 ; break ; 
    }


    LOG(info) << "Rdr::address " << std::setw(10) << getShaderTag() <<  " name " << name << " type " << vnpy->getType() ;

    GLuint       index = location  ;       //  generic vertex attribute to be modified
    GLint         size = vnpy->getSize() ; //  number of components per generic vertex attribute, must be 1,2,3,4
    GLboolean     norm = vnpy->getNorm() ; 
    GLsizei       stride = vnpy->getStride();  ;         // byte offset between consecutive generic vertex attributes, or 0 for tightly packed
    const GLvoid* offset = (const GLvoid*)vnpy->getOffset() ;      

    // offset of the first component of the first generic vertex attribute 
    // in the array in the data store of the buffer currently bound to GL_ARRAY_BUFFER target

    if( vnpy->getIatt() )
    {
        glVertexAttribIPointer(index, size, type, stride, offset);
    }
    else
    {
        glVertexAttribPointer(index, size, type, norm, stride, offset);
    }
    glEnableVertexAttribArray(index);

}






void Rdr::check_uniforms()
{
    bool required = false ; 
    m_mvp_location = m_shader->uniform("ModelViewProjection", required) ; 
    m_mv_location = m_shader->uniform("ModelView", required );     
    m_isnorm_mvp_location = m_shader->uniform("ISNormModelViewProjection", required );     
    m_selection_location = m_shader->uniform("Selection", required );     
    m_flags_location = m_shader->uniform("Flags", required );     
    m_pick_location = m_shader->uniform("Pick", required );     
    m_param_location = m_shader->uniform("Param", required );     
    m_timedomain_location = m_shader->uniform("TimeDomain", required );     
    m_colordomain_location = m_shader->uniform("ColorDomain", required );     
    m_colors_location = m_shader->uniform("Colors", required );     
    m_recselect_location = m_shader->uniform("RecSelect", required );     

    // the "tag" argument of the Rdr identifies the GLSL code being used
    // determining which uniforms are required 

    // TODO: more explicit control of which pipelines need which uniforms ?
    //       currently using optional for everything 
}


void Rdr::update_uniforms()
{

    if(m_composition)
    {
        m_composition->update() ;

        glUniformMatrix4fv(m_mv_location, 1, GL_FALSE,  m_composition->getWorld2EyePtr());
        glUniformMatrix4fv(m_mvp_location, 1, GL_FALSE, m_composition->getWorld2ClipPtr());
        glUniformMatrix4fv(m_isnorm_mvp_location, 1, GL_FALSE, m_composition->getWorld2ClipISNormPtr());

        glm::ivec4 sel = m_composition->getSelection();
        glUniform4i(m_selection_location, sel.x, sel.y, sel.z, sel.w  );    

        glm::ivec4 flg = m_composition->getFlags();
        glUniform4i(m_flags_location, flg.x, flg.y, flg.z, flg.w  );    

        glm::ivec4 pick = m_composition->getPick();
        glUniform4i(m_pick_location, pick.x, pick.y, pick.z, pick.w  );    

        glm::vec4 par = m_composition->getParam();
        glUniform4f(m_param_location, par.x, par.y, par.z, par.w  );    

        glm::vec4 td = m_composition->getTimeDomain();
        glUniform4f(m_timedomain_location, td.x, td.y, td.z, td.w  );    

        glm::vec4 cd = m_composition->getColorDomain();
        glUniform4f(m_colordomain_location, cd.x, cd.y, cd.z, cd.w  );    

        if(m_recselect_location > -1)
        {
            glm::ivec4 recsel = m_composition->getRecSelect();
            //print(recsel, "Rdr::update_uniforms");
            //printf("Rdr::update_uniforms %s  m_recselect_location %d \n", getShaderTag(), m_recselect_location);
            glUniform4i(m_recselect_location, recsel.x, recsel.y, recsel.z, recsel.w  );    
        }

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
    // feed vertices into the pipeline

    glUseProgram(m_program);

    update_uniforms();

    glBindVertexArray(m_vao);

    GLint   first_ = first  ;                            // starting index in the enabled arrays

    GLsizei count_ = count ? count : m_countdefault  ;   // number of indices to be rendered


    glDrawArrays( m_primitive, first_, count_ );


    glBindVertexArray(0);

    glUseProgram(0);
}



void Rdr::dump_uniforms()
{
    LOG(info) << "Rdr::dump_uniforms "
              << " mvp " << m_mvp_location
              << " mv " << m_mv_location 
              << " sel " << m_selection_location 
              << " flg " << m_flags_location 
              << " param " << m_param_location 
              << " isnorm_mvp " << m_isnorm_mvp_location 
              ;
}




