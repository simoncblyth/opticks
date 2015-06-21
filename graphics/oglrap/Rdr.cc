#include <GL/glew.h>

#include "Device.hh"
#include "Rdr.hh"
#include "Prog.hh"
#include "Composition.hh"

// npy-
#include "NPY.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"
#include "stringutil.hpp"

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

    make_shader();  // need to compile and link shader for access to attribute locations

    glUseProgram(m_program);

    check_uniforms();
    
    unsigned int count(0);
    NPYBase* npy(NULL);

    for(unsigned int i=0 ; i<mvn->getNumVecs() ; i++)
    {
        ViewNPY* vnpy = (*mvn)[i] ;
        if(npy == NULL)
        {
            count = vnpy->getCount();
            setCountDefault(count);
            npy = vnpy->getNPY(); 

            upload(npy);
        }
        else
        {
            assert(npy == vnpy->getNPY());     
            LOG(debug) << "Rdr::upload counts, prior: " << count << " current: " << vnpy->getCount() ; 
            assert(count == vnpy->getCount());
        } 
        address(vnpy); 
    }
}


void Rdr::upload(NPYBase* npy)
{
    // handles case of multiple mvn referring to the same buffer without data duplication,
    // by maintaining a list of NPYBase which have been uploaded to the Device

    if(m_device->isUploaded(npy))
    {
        GLuint buffer_id = npy->getBufferId();
        LOG(debug) << "Rdr::upload skip, already uploaded to device : attach to preexisting buffer  " << buffer_id  ;
        attach(buffer_id);
    }
    else
    {
        LOG(debug) << "Rdr::upload " ;

        if(!m_colors_uploaded)
        {
            upload_colors();
            m_colors_uploaded = true ; 
        }


        upload(npy->getBytes(), npy->getNumBytes(0));

        npy->setBufferId(m_buffer); 
        m_device->add(npy);
    }
}


void Rdr::attach(GLuint buffer_id)
{
    // attach to preexisting buffer 
    glGenVertexArrays (1, &m_vao); 
    glBindVertexArray (m_vao);     

    m_buffer = buffer_id ; 
    glBindBuffer(GL_ARRAY_BUFFER, m_buffer);
}



void Rdr::upload(void* data, unsigned int nbytes)
{
    glGenVertexArrays (1, &m_vao); 
    glBindVertexArray (m_vao);     

    glGenBuffers(1, &m_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, m_buffer);
    glBufferData(GL_ARRAY_BUFFER, nbytes, data, GL_STATIC_DRAW );
}


void Rdr::upload_colors()
{
    if(!m_colorbuffer)
    {
         LOG(info) <<"Rdr::upload_colors (" << getShaderTag() << ") no colorbuffer skipping " ;  
         return ; 
    }

    unsigned int ncol = m_colorbuffer->getNumItems() ;
    unsigned char* colors = (unsigned char*)m_colorbuffer->getPointer();
    LOG(info) <<"Rdr::upload_colors (" << getShaderTag() << ") ncol " << ncol ;  

    // https://open.gl/textures
    GLenum  target = GL_TEXTURE_1D ;   // Must be GL_TEXTURE_1D or GL_PROXY_TEXTURE_1D

    glGenTextures(1, &m_colors);
    glBindTexture(target, m_colors);

    GLint   level = 0 ;                // level-of-detail number, Level 0 is the base image level
    GLint   internalFormat = GL_RGBA  ;  // number of color components in the texture
    GLsizei width = ncol ;                // width of the texture image including the border if any (powers of 2 are better)
    GLint   border = 0 ;               // width of the border. Must be either 0 or 1
    GLenum  format = GL_RGBA ;         // format of the pixel data
    GLenum  type = GL_UNSIGNED_BYTE ;  // type of the pixel data

    glTexImage1D(target, level, internalFormat, width, border, format, type, colors );
    delete colors ; 

    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    /*
       With ncol = 5 and using WRAP_S: GL_CLAMP_TO_EDGE, MIN_FILTER : GL_NEAREST, MAG_FILTER : GL_NEAREST
       With colors  R G M Y C  scanned float values to probe the texture algo find...

      
        -0.10  R
     ---------------
         0.00  R 
         0.200 R
     ---------------
         0.201 G 
         0.205 G 
         ...
         0.400 G  
     ---------------
         0.401 M
         0.5   M
         0.6   M 
     ---------------
         0.601 Y
         0.7   Y
         0.8   Y
     --------------
         0.801 C
         0.9   C
         1.0   C
     --------------
         1.1   C
         1.5   C    

   So an n+1 binning approach is used...

In [2]: np.linspace(0.,1.,5+1 )
Out[2]: array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ])
                RRRRRRRRGGGGGGMMMMMM
 
 So to pick via an integer would need 

In [10]: (np.arange(0,5) + 0.5)/5.              (float(i) + 0.5)/5.0  lands mid-bin
Out[10]: array([ 0.1,  0.3,  0.5,  0.7,  0.9])


   */


}



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


    LOG(debug) << "Rdr::address name " << name << " type " << vnpy->getType() ;

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
    m_colors_location = m_shader->uniform("Colors", required );     

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




