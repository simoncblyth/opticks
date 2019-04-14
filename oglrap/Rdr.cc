#include <cstdint>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "PLOG.hh"
#include "BStr.hh"

// npy-
#include "NGLM.hpp"
#include "NGPU.hpp"
#include "NPY.hpp"
#include "ViewNPY.hpp"
#include "MultiViewNPY.hpp"
#include "GLMPrint.hpp"

// optickscore-
#include "Composition.hh"
#include "OpticksEvent.hh"
#include "OpticksBufferControl.hh"


#include <GL/glew.h>

// oglrap-
#include "Device.hh"
#include "Rdr.hh"
#include "Prog.hh"
#include "G.hh"


const char* Rdr::PRINT = "print" ; 


Rdr::Rdr(Device* device, const char* tag, const char* dir, const char* incl_path)
    :
    RendererBase(tag, dir, incl_path),  
    m_first_upload(true),
    m_device(device),
    m_vao(0),
    m_vao_generated(false),
    //m_buffer(0),
    m_countdefault(0),
    m_composition(NULL),
    m_mv_location(-1),
    m_mvp_location(-1),
    m_p_location(-1),
    m_isnorm_mvp_location(-1),
    m_selection_location(-1),
    m_flags_location(-1),
    m_pick_location(-1),
    m_param_location(-1),
    m_nrmparam_location(-1),
    m_scanparam_location(-1),
    m_timedomain_location(-1),
    m_colordomain_location(-1),
    m_colors_location(-1),
    m_recselect_location(-1),
    m_colorparam_location(-1),
    m_lightposition_location(-1),
    m_pickphoton_location(-1),
    m_primitive(GL_POINTS)
{
}


template <typename T>
void Rdr::download( NPY<T>* npy )
{
    OpticksBufferControl ctrl(npy->getBufferControlPtr());
    if(ctrl.isSet(OpticksBufferControl::OPTIX_NON_INTEROP_))
    {
        LOG(info) << "Rdr::download SKIP for " << npy->getBufferName() << " as " << OpticksBufferControl::OPTIX_NON_INTEROP_  ;
        return ; 
    }

    GLenum target = GL_ARRAY_BUFFER ;
    void* ptr = mapbuffer( npy->getBufferId(), target );
    if(ptr)
    {
       npy->read(ptr);
       unmapbuffer(target);
    }
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
void Rdr::setCountDefault(unsigned int count)
{
    m_countdefault = count ;
}
unsigned int Rdr::getCountDefault()
{
    return m_countdefault ;
}
void Rdr::setComposition(Composition* composition)
{
    m_composition = composition ;
}
Composition* Rdr::getComposition()
{
    return m_composition ;
}




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




void Rdr::dump_uploads_table(const char* msg)
{
    const char* tag = getShaderTag();
    LOG(info) << msg << " Rdr tag: " << tag ; 
    typedef std::vector<MultiViewNPY*> VMVN ; 
    for(VMVN::const_iterator it=m_uploads.begin() ; it != m_uploads.end() ; it++)
    {
        MultiViewNPY* mvn = *it ;
        const char* name = mvn->getName() ;
        unsigned int nvec = mvn->getNumVecs() ;
        for(unsigned int i=0 ; i < nvec ; i++)
        {
           ViewNPY* vnpy = (*mvn)[i] ;
           NPYBase*  npy = vnpy->getNPY();

           LOG(info)
              << std::setw(15) << name 
              << std::setw(2) << i << "/" 
              << std::setw(2) << nvec
              << " vnpy " 
              << std::setw(10) << vnpy->getName() 
              << std::setw(10) << vnpy->getCount()
              << " npy "
              << npy->getShapeString()
              << " npy.hasData "
              << npy->hasData()
              ;
        }
    }
}


void Rdr::upload(MultiViewNPY* mvn, bool debug)
{

    if(!mvn) return ; 

    m_uploads.push_back(mvn);

    // MultiViewNPY are constrained to all refer to the same underlying NPY 
    // so only do upload and m_buffer creation for the first 

    const char* tag = getShaderTag();
   
    if(debug)
    {
        LOG(info) << "Rdr::upload tag [" << tag << "] mvn [" << mvn->getName() << "]" ; 
        mvn->Summary("Rdr::upload mvn");
    }

    // need to compile and link shader for access to attribute locations
    if(m_first_upload)
    {
        prepare_vao(); // seems needed by oglrap-/tests/AxisTest 
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
                if(debug)
                LOG(info) << "Rdr::upload" 
                          << " mvn " << mvn->getName() 
                          << " (first)count " << count
                           ;
                setCountDefault(count);
            }
            else
            {
                bool count_match = count == getCountDefault() ;
                if(!count_match)
                {
                    LOG(fatal) << "Rdr::upload COUNT MISMATCH " 
                               << " tag " << tag 
                               << " mvn " << mvn->getName() 
                               << " expected  " << getCountDefault()
                               << " found " << count 
                               ; 
                    dump_uploads_table();
                }
                assert(count_match && "all buffers fed to the Rdr pipeline must have the same counts");
            }

            npy = vnpy->getNPY(); 
            upload(npy, vnpy);      // duplicates are not re-uploaded
        }
        else
        {
            assert(npy == vnpy->getNPY());     
            LOG(verbose) << "Rdr::upload counts, prior: " << count << " current: " << vnpy->getCount() ; 
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
    LOG(debug)
                 << "Rdr::log " 
                 << std::setw(10) << getShaderTag() 
                 << " "
                 << msg  
                 << value ;
 
}


void Rdr::prepare_vao()
{
    G::ErrCheck("Rdr::prepare_vao.[", true);
    if(!m_vao_generated)
    {
        glGenVertexArrays (1, &m_vao); 
        m_vao_generated = true ; 
        log("prepare_vao : generate m_vao:", m_vao);
   }

    log("prepare_vao : bind m_vao:", m_vao);


    glBindVertexArray (m_vao);     
    G::ErrCheck("Rdr::prepare_vao.]", true);

}


void Rdr::upload(NPYBase* npy, ViewNPY* vnpy)
{
    // handles case of multiple mvn referring to the same buffer without data duplication,
    // by maintaining a list of NPYBase which have been uploaded to the Device

    prepare_vao();

    MultiViewNPY* parent = vnpy->getParent();
    assert(parent);

    bool dynamic = npy->isDynamic();

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

        char repdata[16] ;
        snprintf( repdata, 16, "%p", data );

        GLuint buffer_id ;  
        glGenBuffers(1, &buffer_id);
        glBindBuffer(GL_ARRAY_BUFFER, buffer_id);

        LOG(debug) 
                  << std::setw(15) << parent->getName() 
                  << std::setw(5)  << vnpy->getName()
                  << " cn " << std::setw(8) << vnpy->getCount()
                  << " sh " << std::setw(20) << vnpy->getShapeString()
                  << " id " << std::setw(5) << buffer_id
                  << " dt " << std::setw(16) << repdata 
                  << " hd " << std::setw(5) << ( npy->hasData() ? "Y" : "N" )
                  << " nb " << std::setw(10) << nbytes 
                  << " " << (dynamic ? "GL_DYNAMIC_DRAW" : "GL_STATIC_DRAW" )
                  ;   

        glBufferData(GL_ARRAY_BUFFER, nbytes, data, dynamic ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW  );

        npy->setBufferId(buffer_id); 
        m_device->add(npy);         //  (void*)npy used by Device::isUploaded to prevent re-uploads  


        NGPU::GetInstance()->add( nbytes, vnpy->getName(), parent->getName(), "Rdr:upl" ); 
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

    GLenum type = GL_FLOAT  ;              //  of each component in the array
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


    GLuint       index = location  ;            //  generic vertex attribute to be modified
    GLint         size = vnpy->getSize() ;      //  number of components per generic vertex attribute, must be 1,2,3,4
    GLboolean     norm = vnpy->getNorm() ; 
    GLsizei       stride = vnpy->getStride();   // byte offset between consecutive generic vertex attributes, or 0 for tightly packed

    uintptr_t stride_ = stride ;
    uintptr_t offset_ = vnpy->getOffset() ;

    const GLvoid* offset = (const GLvoid*)offset_ ;      

    // offset of the first component of the first generic vertex attribute 
    // in the array in the data store of the buffer currently bound to GL_ARRAY_BUFFER target

    LOG(verbose) << "Rdr::address (glVertexAttribPointer) "
              << std::setw(10) << getShaderTag() 
              << " name " << name 
              << " type " << std::setw(20) << vnpy->getTypeName() 
              << " index " << index
              << " norm " << norm
              << " size " << size
              << " stride " << stride
              << " offset_ " << offset_
              ;

    assert( offset_ < stride_ && "offset_ should always be less than the stride_, see ggv-/issues/gui_broken_photon_record_colors");

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
    m_p_location = m_shader->uniform("Projection", required );     
    m_isnorm_mvp_location = m_shader->uniform("ISNormModelViewProjection", required );     
    m_selection_location = m_shader->uniform("Selection", required );     
    m_flags_location = m_shader->uniform("Flags", required );     
    m_pick_location = m_shader->uniform("Pick", required );     
    m_param_location = m_shader->uniform("Param", required );     

    m_nrmparam_location = m_shader->uniform("NrmParam",  required); 
    m_scanparam_location = m_shader->uniform("ScanParam",  required); 

    m_timedomain_location = m_shader->uniform("TimeDomain", required );     
    m_colordomain_location = m_shader->uniform("ColorDomain", required );     
    m_colors_location = m_shader->uniform("Colors", required );     
    m_recselect_location = m_shader->uniform("RecSelect", required );     
    m_pickphoton_location = m_shader->uniform("PickPhoton", required );     
    m_colorparam_location = m_shader->uniform("ColorParam", required );     
    m_lightposition_location = m_shader->uniform("LightPosition", required); 

    // the "tag" argument of the Rdr identifies the GLSL code being used
    // determining which uniforms are required 

    // TODO: more explicit control of which pipelines need which uniforms ?
    //       currently using optional for everything 
}


void Rdr::update_uniforms()
{

    if(m_composition)
    {
        // m_composition->update() ; moved up to Scene::render

        glUniformMatrix4fv(m_mv_location, 1, GL_FALSE,  m_composition->getWorld2EyePtr());

        glUniformMatrix4fv(m_mvp_location, 1, GL_FALSE, m_composition->getWorld2ClipPtr());

        glUniformMatrix4fv(m_p_location, 1, GL_FALSE, m_composition->getProjectionPtr());

        glUniformMatrix4fv(m_isnorm_mvp_location, 1, GL_FALSE, m_composition->getWorld2ClipISNormPtr());

        glm::ivec4 sel = m_composition->getSelection();
        glUniform4i(m_selection_location, sel.x, sel.y, sel.z, sel.w  );    

        glm::ivec4 flg = m_composition->getFlags();
        glUniform4i(m_flags_location, flg.x, flg.y, flg.z, flg.w  );    

        glm::ivec4 pick = m_composition->getPick();
        glUniform4i(m_pick_location, pick.x, pick.y, pick.z, pick.w  );    

        glm::vec4 par = m_composition->getParam();
        glUniform4f(m_param_location, par.x, par.y, par.z, par.w  );    


        glUniform4fv(m_scanparam_location, 1, m_composition->getScanParamPtr());
        //glm::vec4 sp = m_composition->getScanParam(); 

        glm::ivec4 np = m_composition->getNrmParam(); 
        glUniform4i(m_nrmparam_location, np.x, np.y, np.z, np.w);

        //LOG(info) << "Rdr::update_uniforms"
        //          << " NrmParam " << gformat(np)
        //          << " ScanParam " << gformat(sp)
        //           ;



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

        if(m_pickphoton_location > -1)
        {
            glm::ivec4 pp = m_composition->getPickPhoton();
            glUniform4i(m_pickphoton_location, pp.x, pp.y, pp.z, pp.w  );    
        }

        if(m_colorparam_location > -1)
        {
            glm::ivec4 colpar = m_composition->getColorParam();
            glUniform4i(m_colorparam_location, colpar.x, colpar.y, colpar.z, colpar.w  );    
        }




        glUniform4fv(m_lightposition_location, 1, m_composition->getLightPositionPtr());




    } 
    else
    { 
        assert(0 && "Rdr without composition");
        glm::mat4 identity ; 
        glUniformMatrix4fv(m_mv_location, 1, GL_FALSE, glm::value_ptr(identity));
        glUniformMatrix4fv(m_mvp_location, 1, GL_FALSE, glm::value_ptr(identity));
        glUniformMatrix4fv(m_p_location, 1, GL_FALSE, glm::value_ptr(identity));
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
              << " p " << m_p_location 
              << " sel " << m_selection_location 
              << " flg " << m_flags_location 
              << " param " << m_param_location 
              << " isnorm_mvp " << m_isnorm_mvp_location 
              ;
}



void Rdr::download( OpticksEvent* evt )
{
    NPY<float>* ox = evt->getPhotonData();
    if(ox)
        Rdr::download(ox);

    NPY<short>* rx = evt->getRecordData();
    if(rx)
        Rdr::download(rx);

    NPY<unsigned long long>* ph = evt->getSequenceData();
    if(ph)
        Rdr::download(ph);

}


template OGLRAP_API void Rdr::download<float>(NPY<float>*);

