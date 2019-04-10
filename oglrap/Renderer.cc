#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdint>

// brap-
#include "BBufSpec.hh"

// npy-
#include "NGLM.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
#include "NSlice.hpp"

#include "Opticks.hh"


#include <GL/glew.h>



#include "RBuf.hh"
#include "RBuf4.hh"

#include "Renderer.hh"
#include "InstLODCull.hh"
#include "Prog.hh"
#include "Composition.hh"
#include "Texture.hh"

// ggeo
#include "GArray.hh"
#include "GBuffer.hh"
#include "GMergedMesh.hh"
#include "GBBoxMesh.hh"
#include "GDrawable.hh"

#include "PLOG.hh"




struct DrawElements
{
    DrawElements( GLenum mode_ , GLsizei count_ , GLenum  type_ , void* indices_ , GLsizei  primcount_ )
        :
        mode(mode_),
        count(count_),
        type(type_),
        indices(indices_),
        primcount(primcount_)
    {}

    GLenum  mode  ; 
    GLsizei count ; 
    GLenum  type ; 
    void*  indices ; 
    GLsizei  primcount ;   // only for Instanced

    std::string desc() const 
    {
        std::stringstream ss ; 
        ss << "DrawElements"
           << " count " << std::setw(10) << count 
           << " type " << std::setw(10) << type
           << " indices " << std::setw(10) << indices
           << " primcount " << std::setw(10) << primcount 
           ;
        return ss.str(); 
    }

};



const char* Renderer::PRINT = "print" ; 


Renderer::Renderer(const char* tag, const char* dir, const char* incl_path)
    :
    RendererBase(tag, dir, incl_path),
    m_tagtex(strcmp(tag,"tex")==0),
    m_draw_0(0),
    m_draw_1(1),

    m_vbuf(NULL),
    m_nbuf(NULL),
    m_cbuf(NULL),
    m_tbuf(NULL),
    m_fbuf(NULL),
    m_ibuf(NULL),

    m_dst(NULL),
#ifdef QUERY_WORKAROUND
    m_dst_devnull(NULL),
#endif

    m_mv_location(-1),
    m_mvp_location(-1),
    m_clip_location(-1),
    m_param_location(-1),
    m_scanparam_location(-1),
    m_nrmparam_location(-1),
    m_lightposition_location(-1),
    m_itransform_location(-1),
    m_colordomain_location(-1),
    m_colors_location(-1),
    m_pickface_location(-1),
    m_colorTex_location(-1),
    m_depthTex_location(-1),

    m_itransform_count(0),
    m_draw_count(0),
    m_indices_count(0),
    m_drawable(NULL),
    m_geometry(NULL),
    m_bboxmesh(NULL),
    m_texture(NULL),
    m_texture_id(-1),
    m_composition(NULL),
    m_has_tex(false),
    m_has_transforms(false),
    m_instanced(false),
    m_wireframe(false),

    m_instlodcull(NULL),
    m_instlodcull_enabled(false),

    m_num_lod(-1),
    m_test_lod(0),
    m_use_lod(true), 
    m_lod(0),

    m_type(NULL)
{

    for(unsigned i=0 ; i < MAX_LOD ; i++) m_lod_counts[i] = 0 ; 
}


Renderer::~Renderer()
{
}

void Renderer::setType(const char* type)
{
    m_type = type ; 
}


bool Renderer::isInstLODCullEnabled() const 
{
   return m_instlodcull_enabled ;
}


void Renderer::setLOD(int lod)
{
    m_lod = lod ; 
}

void Renderer::setNumLOD(int num_lod)
{
    m_num_lod = num_lod ; 
}


void Renderer::setInstanced(bool instanced)
{
    m_instanced = instanced ; 
}
void Renderer::setInstLODCull(InstLODCull* instlodcull)
{
    assert(m_instanced);
    m_instlodcull = instlodcull ; 
}

void Renderer::setWireframe(bool wireframe)
{
    m_wireframe = wireframe ; 
}
void Renderer::setComposition(Composition* composition)
{
    m_composition = composition ;
}
Composition* Renderer::getComposition()
{
    return m_composition ;
}

void Renderer::configureI(const char* name, std::vector<int> values )
{
    if(values.empty()) return ; 
    if(strcmp(name, PRINT)==0) Print("Renderer::configureI");
}

//////////  CPU side buffer setup  ///////////////////

std::string Renderer::brief() const 
{
    std::stringstream ss ; 
    ss << " Renderer "
       << " tag " << getShaderTag()
       << " type " << ( m_type ? m_type : "-" )
       << " idx " << ( m_drawable ? m_drawable->getIndex() : -1 )
       ;
    return ss.str();
}


std::string Renderer::desc() const 
{
    std::stringstream ss ; 
    ss << brief()    
       << " instlodcull " << ( m_instlodcull ? "YES" : "NO" )
       << " lod " << m_lod 
       << " num_lod " << m_num_lod 
       << " m_itransform_count " << m_itransform_count
       << " lod_counts "
       << " (" 
       << std::setw(7) << m_lod_counts[0] << " "
       << std::setw(7) << m_lod_counts[1] << " "
       << std::setw(7) << m_lod_counts[2] << " "
       << ")"
       << " tot " 
       << std::setw(7) << m_lod_counts[0]+m_lod_counts[1]+m_lod_counts[2]
       ;

    return ss.str();
}

const char* Renderer::GBBoxMesh_ = "GBBoxMesh" ; 
const char* Renderer::GMergedMesh_ = "GMergedMesh" ; 
const char* Renderer::Texture_ = "Texture" ; 

void Renderer::upload(GBBoxMesh* bboxmesh)
{
    m_bboxmesh = bboxmesh ;
    assert( m_geometry == NULL && m_texture == NULL );  // exclusive 
    setType(GBBoxMesh_);

    setDrawable(bboxmesh);
    upload();
    setupDraws(NULL) ; 
}
void Renderer::upload(GMergedMesh* mm)
{
    m_geometry = mm ;
    unsigned num_comp = mm->getNumComponents();
    setNumLOD(num_comp);

    setType(GMergedMesh_);

    assert( m_texture == NULL && m_bboxmesh == NULL );  // exclusive 

    setDrawable(mm);
    upload();
    setupDraws(mm);
}
void Renderer::upload(Texture* texture)
{
    assert( m_geometry == NULL && m_bboxmesh == NULL );  // exclusive 
    setType(Texture_);

    m_texture = texture ;
    m_texture_id = texture->getId();

    setDrawable(texture);
    upload();
    setupDraws(NULL) ; 
}


void Renderer::setDrawable(GDrawable* drawable) // CPU side buffer setup
{
    assert(drawable);
    m_drawable = drawable ;

    NSlice* islice = drawable->getInstanceSlice();
    NSlice* fslice = drawable->getFaceSlice();
 
    //  nvert: vertices, normals, colors
    m_vbuf = MAKE_RBUF(m_drawable->getVerticesBuffer());
    m_nbuf = MAKE_RBUF(m_drawable->getNormalsBuffer());
    m_cbuf = MAKE_RBUF(m_drawable->getColorsBuffer());

    assert(m_vbuf->getNumBytes() == m_cbuf->getNumBytes());
    assert(m_nbuf->getNumBytes() == m_cbuf->getNumBytes());
 
    // 3*nface indices
    GBuffer* fbuf_orig = m_drawable->getIndicesBuffer();
    GBuffer* fbuf = fslice ? fslice_element_buffer(fbuf_orig, fslice) : fbuf_orig ;

    m_fbuf = MAKE_RBUF(fbuf) ;
    
    m_tbuf = MAKE_RBUF(m_drawable->getTexcoordsBuffer());
    setHasTex(m_tbuf != NULL);

    NPY<float>* ibuf_orig = m_drawable->getITransformsBuffer();

    if(islice)
        LOG(warning) << "Renderer::setDrawable instance slicing ibuf with " << islice->description() ;

    NPY<float>* ibuf = islice ? ibuf_orig->make_slice(islice) :  ibuf_orig ;
    if(ibuf) ibuf->setName("itransforms"); 

    m_ibuf = MAKE_RBUF(ibuf) ; 
    setHasTransforms(m_ibuf != NULL);
}



void Renderer::upload()
{
    unsigned num_instances = m_ibuf ? m_ibuf->getNumItems() : 0 ;

    m_instlodcull_enabled = m_instlodcull && m_num_lod > 0 && num_instances > InstLODCull::INSTANCE_MINIMUM ;
    // Renderer::upload(GMergedMesh*) sets num_lod from mm components, >0 only for instanced mm
      
    // (June 2018) IS THIS DUPLICATING THE UPLOADS WITH THE m_vao_all
    //m_vao_all = createVertexArray(m_ibuf);   // DEBUGGING ONLY 

    RBuf::Owner = strdup(getName()) ;  

    if(m_instlodcull_enabled) 
    {
        createVertexArrayLOD(); 
    } 
    else if(m_lod < 0)
    {
        // (June 2018) moved this here from outside the if to avoid duplicating the vertices
        m_vao_all = createVertexArray(m_ibuf);   // DEBUGGING ONLY 
    } 
    else
    {
        m_vao[0] = createVertexArray(m_ibuf);
    }

    make_shader();  // requires VAO bound to pass validation

    glUseProgram(m_program);  // moved prior to check uniforms following Rdr::upload

    glEnable(GL_CLIP_DISTANCE0); 
 
    check_uniforms();
}


void Renderer::createVertexArrayLOD()
{
    // Initially tried cloning and uploading of forked buffers prior
    // to VAO creation, this didnt fly ... got the usual black screen for instanced renders
    // Instead rejig to keep all OpenGL setup within the VAO "context".
    //
    // Buf cloning/forking prior to that just copies buffer vital stats
    // and sets the gpu_resident property to control which buffers need the upload 

    assert( m_instlodcull_enabled ); 
 
    //int debug_clone_slot = m_test_lod ; // <-- actually clone and mark as **NOT** GPU resident : SO **WILL** BE UPLOADED (FOR DEBUGGING PRIOR TO TXF OPERATIONAL)
    int debug_clone_slot = -1 ; //  -1 : relying on txf to populate all the gpu buffers

    m_dst = RBuf4::MakeFork(m_ibuf, m_num_lod, debug_clone_slot  );

    for(int i=0 ; i < m_num_lod ; i++) m_vao[i] = createVertexArray(m_dst->at(i));   // vao for rendering with the derived instance transforms

#ifdef QUERY_WORKAROUND
    unsigned num_bytes = 1 ; 
    m_dst_devnull = RBuf4::MakeDevNull(m_num_lod, num_bytes );
    m_dst_devnull->uploadNull(GL_ARRAY_BUFFER, GL_DYNAMIC_COPY );
    m_instlodcull->setupFork(m_ibuf, m_dst, m_dst_devnull );
#else
    m_instlodcull->setupFork(m_ibuf, m_dst, NULL );
#endif

}


GLuint Renderer::createVertexArray(RBuf* instanceBuffer)
{
    /*
     With ICDemo do VAO creation after uploading all buffers... 
     somehow that doesnt work here (gives unexpected "broken" render of instances)
     perhaps because there are multiple renderers ? Anyhow doesnt matter as uploads are not-redone.

     * as multiple GL_ARRAY_BUFFER in use must bind the appropriate ones prior to glVertexAttribPointer
       in order to capture the (buffer,attrib) "coordinates" into the VAO

     * getNumElements gives 3 for both vertex and color items

     * enum values vPosition, vNormal, vColor, vTexcoord are duplicating layout numbers in the nrm/vert.glsl  

     Without glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, m_indices);
     got a blank despite being bound in the upload 
     when VAO creation was after upload. 
    
     VAO creation needs to be before the uploads in order for it 
     to capture this state.
    
     * although there is only a single GL_ELEMENT_ARRAY_BUFFER 
       within this context recall that there are multiple Renderers and Rdr 
       sharing the same OpenGL context so it is necessary to repeat the binding  
     
     TODO: consider adopting the more flexible ViewNPY approach used for event data    
    */

    GLuint vao ; 
    glGenVertexArrays (1, &vao); 
    glBindVertexArray (vao);     

    if(instanceBuffer) 
    {
        if(instanceBuffer->gpu_resident)
        {
            assert( m_instlodcull_enabled ); 
            instanceBuffer->uploadNull(GL_ARRAY_BUFFER, GL_DYNAMIC_COPY );
        }
        else
        {
            instanceBuffer->upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW );
        } 
    }

    GLuint instanceBO = instanceBuffer ? instanceBuffer->getBufferId() : 0u ;
    m_itransform_count = instanceBuffer ? instanceBuffer->getNumItems() : 0 ;

    if(m_verbosity > 3)
    std::cout << "Renderer::createVertexArray"
              << " vao " << vao 
              << " itransform_count " << m_itransform_count 
              << " instanceBO  " << instanceBO  
              << " desc " << desc()
              << std::endl
               ;


    GLboolean normalized = GL_FALSE ; 
    GLsizei stride = 0 ;
    const GLvoid* offset = NULL ;
 
    if(instanceBO > 0)
    {
        LOG(verbose) << "Renderer::upload_buffers setup instance transform attributes " ;
        glBindBuffer (GL_ARRAY_BUFFER, instanceBO);

        uintptr_t qsize = sizeof(GLfloat) * 4 ;
        GLsizei matrix_stride = qsize * 4 ;

        glVertexAttribPointer(vTransform + 0 , 4, GL_FLOAT, normalized, matrix_stride, (void*)0 );
        glVertexAttribPointer(vTransform + 1 , 4, GL_FLOAT, normalized, matrix_stride, (void*)(qsize));
        glVertexAttribPointer(vTransform + 2 , 4, GL_FLOAT, normalized, matrix_stride, (void*)(qsize*2));
        glVertexAttribPointer(vTransform + 3 , 4, GL_FLOAT, normalized, matrix_stride, (void*)(qsize*3));

        glEnableVertexAttribArray (vTransform + 0);   
        glEnableVertexAttribArray (vTransform + 1);   
        glEnableVertexAttribArray (vTransform + 2);   
        glEnableVertexAttribArray (vTransform + 3);   

        GLuint divisor = 1 ;   // number of instances between updates of attribute , >1 will land that many instances on top of each other
        glVertexAttribDivisor(vTransform + 0, divisor);  // dictates instanced geometry shifts between instances
        glVertexAttribDivisor(vTransform + 1, divisor);
        glVertexAttribDivisor(vTransform + 2, divisor);
        glVertexAttribDivisor(vTransform + 3, divisor);
    } 

    // NB already uploaded buffers are just bind not uploaded again

    m_vbuf->upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    m_cbuf->upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    m_nbuf->upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW);

    if(m_tbuf) m_tbuf->upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW);

    m_fbuf->upload(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
    m_indices_count = m_fbuf->getNumItems(); // number of indices, would be 3 for a single triangle

    glBindBuffer (GL_ARRAY_BUFFER, m_vbuf->getBufferId() );
    glVertexAttribPointer(vPosition, m_vbuf->getNumElements(), GL_FLOAT, normalized, stride, offset);
    glEnableVertexAttribArray (vPosition);  

    glBindBuffer (GL_ARRAY_BUFFER, m_nbuf->getBufferId() );
    glVertexAttribPointer(vNormal, m_nbuf->getNumElements(), GL_FLOAT, normalized, stride, offset);
    glEnableVertexAttribArray (vNormal);  

    glBindBuffer (GL_ARRAY_BUFFER, m_cbuf->getBufferId() );
    glVertexAttribPointer(vColor, m_cbuf->getNumElements(), GL_FLOAT, normalized, stride, offset);
    glEnableVertexAttribArray (vColor);   

    if(m_tbuf)
    {
        glBindBuffer (GL_ARRAY_BUFFER, m_tbuf->getBufferId()  );
        glVertexAttribPointer(vTexcoord, m_tbuf->getNumElements(), GL_FLOAT, normalized, stride, offset);
        glEnableVertexAttribArray (vTexcoord);   
    }

    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, m_fbuf->getBufferId() );

    return vao ; 
}




void Renderer::setupDrawsLOD(GMergedMesh* mm)
{
    assert(mm);

    mm->dumpComponents("Renderer::setupDrawsLOD"); 
    
    int num_comp = mm ? mm->getNumComponents() : 0  ; 
    assert( num_comp > 0 ) ;

    for(int i=0 ; i < num_comp ; i++)
    {
        glm::uvec4 eidx ;
        mm->getComponent(eidx, i );

        unsigned offset_face = eidx.x ;
        unsigned num_face = eidx.y ;
        GLsizei count = num_face*3 ; 
        void* offset_indices = (void*)(offset_face*3*sizeof(unsigned)) ; 

        m_draw[i] = new DrawElements( GL_TRIANGLES, count, GL_UNSIGNED_INT, offset_indices, m_itransform_count );
        // indices_count would be 3 for a single triangle, 30 for ten triangles
    }

    if(m_lod > 0)
    {
        m_draw_0 = 0 ; 
        m_draw_1 = num_comp ; 
    }
    else
    {
        m_draw_0 = -m_lod ; 
        m_draw_1 = -m_lod + 1; 
    }

    LOG(warning) << "Renderer::setupDrawsLOD"
                 << " num_comp " << num_comp 
                 << " m_lod " << m_lod 
                 << " m_draw_0 " << m_draw_0
                 << " m_draw_1 " << m_draw_1
                 ;

}



void Renderer::setupDraws(GMergedMesh* mm)
{   
    int num_comp = mm ? mm->getNumComponents() : 0  ; 

    //bool one_draw = mm == NULL || num_comp < 1 || m_instlodcull_enabled == false ;
    bool one_draw = mm == NULL || num_comp < 1 ;

    LOG(debug) << "Renderer::setupDraws"
              << brief()
              << " num_comp " << num_comp 
              << " m_lod " << m_lod 
              << " one_draw " << ( one_draw ? "YES" : "NO" )
              ;

    if(one_draw)
    {
        m_draw[0] = new DrawElements( GL_TRIANGLES, m_indices_count, GL_UNSIGNED_INT, NULL, m_itransform_count );     
        // indices_count would be 3 for a single triangle, 30 for ten triangles
    }
    else
    {
        setupDrawsLOD(mm);
    }
}


void Renderer::cull()
{
    assert( m_instlodcull_enabled );
    m_instlodcull->launch();
}

void Renderer::render()
{ 
    //LOG(info) << "Renderer::render" ; 

    // if(m_instlodcull_enabled) cull();   moved to Scene::preRenderCompute 

    glUseProgram(m_program);

    update_uniforms();

    glActiveTexture(GL_TEXTURE0 + TEX_UNIT_0 );
    glBindTexture(GL_TEXTURE_2D,  m_texture_id );

    // https://www.opengl.org/archives/resources/faq/technical/transparency.htm
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
    glEnable (GL_BLEND);

    if(m_wireframe) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);


    if( m_instlodcull_enabled )
    {
        assert(m_instanced);
        //LOG(info) << "glDrawElementsInstanced.instlodcull_enabled"  ;  

        for(unsigned i=m_draw_0 ; i < m_draw_1 ; i++)
        { 
            glBindVertexArray ( m_use_lod ? m_vao[i] : m_vao_all );

            const DrawElements& draw = *m_draw[i] ;   

            m_lod_counts[i] = m_use_lod ? m_dst->at(i)->query_count : draw.primcount ;

            glDrawElementsInstanced( draw.mode, draw.count, draw.type,  draw.indices, m_lod_counts[i]  ) ;

        }
        if(m_verbosity > 0)
            std::cout << desc() << std::endl ; 
    }
    else if( m_lod < 0)   // debugging LOD rendering 
    {
        for(unsigned i=m_draw_0 ; i < m_draw_1 ; i++)
        { 
            glBindVertexArray ( m_vao_all );

            const DrawElements& draw = *m_draw[i] ;   
            //LOG(info) << "glDrawElementsInstanced.lod<0" << draw.desc() ;  

            m_lod_counts[i] = draw.primcount ;

            glDrawElementsInstanced( draw.mode, draw.count, draw.type,  draw.indices, m_lod_counts[i]  ) ;
        }
    } 
    else
    {
        glBindVertexArray ( m_vao[0] );
        const DrawElements& draw = *m_draw[0] ;   

        if(m_instanced)
        {
            //LOG(info) << "glDrawElementsInstanced " << draw.desc() ;  
            glDrawElementsInstanced( draw.mode, draw.count, draw.type,  draw.indices, draw.primcount  ) ;
        }
        else
        {
            //if(m_tagtex) LOG(info) << "(tagtex) glDrawElements " << draw.desc() ;  
            glDrawElements( draw.mode, draw.count, draw.type,  draw.indices ) ;
        }
    }


    if(m_wireframe) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    m_draw_count += 1 ; 

    glBindVertexArray(0);

    glUseProgram(0);
}










GBuffer* Renderer::fslice_element_buffer(GBuffer* fbuf_orig, NSlice* fslice)
{
    assert(fslice);
    LOG(warning) << "Renderer::fslice_element_buffer face slicing the indices buffer " << fslice->description() ; 

    unsigned nelem = fbuf_orig->getNumElements();
    assert(nelem == 1);

    // temporarily reshape so can fslice at tri/face level 
   
    fbuf_orig->reshape(3);  // equivalent to NumPy buf.reshape(-1,3)  putting 3 triangle indices into each item 
    GBuffer* fbuf = fbuf_orig->make_slice(fslice);
    fbuf_orig->reshape(nelem);   // equivalent to NumPy buf.reshape(-1,1) 

    fbuf->reshape(nelem);        // sliced buffer adopts shape of source, so reshape this too
    assert(fbuf->getNumElements() == 1);
    return fbuf ;  
}



void Renderer::check_uniforms()
{
    std::string tag = getShaderTag();

    bool required = false;

    bool nrm  = tag.compare("nrm") == 0 ; 
    bool nrmvec = tag.compare("nrmvec") == 0 ; 
    bool inrm = tag.compare("inrm") == 0 ; 
    bool tex = tag.compare("tex") == 0 ; 

    LOG(verbose) << "Renderer::check_uniforms " 
              << " tag " << tag  
              << " nrm " << nrm  
              << " nrmvec " << nrmvec  
              << " inrm " << inrm
              << " tex " << tex
              ;  

    assert( nrm ^ inrm ^ tex ^ nrmvec );

    if(nrm || inrm)
    {
        m_mvp_location = m_shader->uniform("ModelViewProjection", required); 
        m_mv_location =  m_shader->uniform("ModelView",           required);      
        m_clip_location = m_shader->uniform("ClipPlane",          required); 
        m_param_location = m_shader->uniform("Param",          required); 
        m_nrmparam_location = m_shader->uniform("NrmParam",         required); 
        m_scanparam_location = m_shader->uniform("ScanParam",         required); 

        m_lightposition_location = m_shader->uniform("LightPosition",required); 

        m_colordomain_location = m_shader->uniform("ColorDomain", required );     
        m_colors_location = m_shader->uniform("Colors", required );     

        if(inrm)
        {
            m_itransform_location = m_shader->uniform("InstanceTransform",required); 
            // huh, aint this an att rather than a uni ?
        } 
    } 
    else if(nrmvec)
    {
        m_mvp_location = m_shader->uniform("ModelViewProjection", required); 
        m_pickface_location = m_shader->uniform("PickFace", required); 
    }
    else if(tex)
    {
        // still being instanciated at least, TODO: check regards this cf the OptiXEngine internal renderer
        m_mv_location =  m_shader->uniform("ModelView",           required);    
        m_colorTex_location = m_shader->uniform("ColorTex", required);
        m_depthTex_location = m_shader->uniform("DepthTex", required);

        m_nrmparam_location = m_shader->uniform("NrmParam",         required); 
        m_scanparam_location = m_shader->uniform("ScanParam",         required); 
        m_clip_location = m_shader->uniform("ClipPlane",          required); 

    } 
    else
    {
        LOG(fatal) << "Renderer::checkUniforms unexpected shader tag " << tag ; 
        assert(0); 
    }

    LOG(verbose) << "Renderer::check_uniforms "
              << " tag " << tag 
              << " mvp " << m_mvp_location
              << " mv " << m_mv_location 
              << " nrmparam " << m_nrmparam_location 
              << " scanparam " << m_scanparam_location 
              << " clip " << m_clip_location 
              << " itransform " << m_itransform_location 
              ;

}

void Renderer::update_uniforms()
{
    if(m_composition)
    {
        //m_composition->update() ;  
        //    moved up to Scene::render repeat this in every renderer ?

        glUniformMatrix4fv(m_mv_location, 1, GL_FALSE,  m_composition->getWorld2EyePtr());
        glUniformMatrix4fv(m_mvp_location, 1, GL_FALSE, m_composition->getWorld2ClipPtr());


        glUniform4fv(m_param_location, 1, m_composition->getParamPtr());

        glUniform4fv(m_scanparam_location, 1, m_composition->getScanParamPtr());
        //glm::vec4 sp = m_composition->getScanParam(); 

        glm::ivec4 np = m_composition->getNrmParam(); 
        glUniform4i(m_nrmparam_location, np.x, np.y, np.z, np.w);

        
/*
        LOG(info) << "Renderer::update_uniforms"
                  << " NrmParam " << gformat(np)
                  << " ScanParam " << gformat(sp)
                   ;
*/

        glUniform4fv(m_lightposition_location, 1, m_composition->getLightPositionPtr());

        glUniform4fv(m_clip_location, 1, m_composition->getClipPlanePtr() );


        glm::vec4 cd = m_composition->getColorDomain();
        glUniform4f(m_colordomain_location, cd.x, cd.y, cd.z, cd.w  );    


        if(m_pickface_location > -1)
        {
            glm::ivec4 pf = m_composition->getPickFace();
            glUniform4i(m_pickface_location, pf.x, pf.y, pf.z, pf.w  );    
        }



        if(m_composition->getClipMode() == -1)
        {
            glDisable(GL_CLIP_DISTANCE0); 
        }
        else
        {
            glEnable(GL_CLIP_DISTANCE0); 
        }

        //if(m_draw_count == 0)
        //    print( m_composition->getClipPlanePtr(), "Renderer::update_uniforms ClipPlane", 4);

    } 
    else
    { 
        LOG(warning) << "Renderer::update_uniforms without composition " ; 

        glm::mat4 identity ; 
        glUniformMatrix4fv(m_mv_location, 1, GL_FALSE, glm::value_ptr(identity));
        glUniformMatrix4fv(m_mvp_location, 1, GL_FALSE, glm::value_ptr(identity));
    }


    if(m_has_tex)
    {
        glUniform1i(m_colorTex_location, TEX_UNIT_0 );
        glUniform1i(m_depthTex_location, TEX_UNIT_1 );
    }
}







void Renderer::dump(RBuf* buf)
{
    dump( buf->getPointer(),buf->getNumBytes(),buf->getNumElements()*sizeof(float),0,buf->getNumItems() );    
}


void Renderer::dump(void* data, unsigned int /*nbytes*/, unsigned int stride, unsigned long offset, unsigned int count )
{
    //assert(m_composition) rememeber OptiXEngine uses a renderer internally to draw the quad texture
    if(m_composition) m_composition->update();

    for(unsigned int i=0 ; i < count ; ++i )
    {
        if(i < 5 || i > count - 5)
        {
            char* ptr = (char*)data + offset + i*stride  ; 
            float* f = (float*)ptr ; 

            float x(*(f+0));
            float y(*(f+1));
            float z(*(f+2));

            if(m_composition)
            {
                glm::vec4 w(x,y,z,1.f);
                glm::mat4 w2e = glm::make_mat4(m_composition->getWorld2EyePtr()); 
                glm::mat4 w2c = glm::make_mat4(m_composition->getWorld2ClipPtr()); 

               // print(w2e, "w2e");
               // print(w2c, "w2c");

                glm::vec4 e  = w2e * w ;
                glm::vec4 c =  w2c * w ;
                glm::vec4 cdiv =  c/c.w ;

                printf("RendererBase::dump %7u/%7u : w(%10.1f %10.1f %10.1f) e(%10.1f %10.1f %10.1f) c(%10.3f %10.3f %10.3f %10.3f) c/w(%10.3f %10.3f %10.3f) \n", i,count,
                        w.x, w.y, w.z,
                        e.x, e.y, e.z,
                        c.x, c.y, c.z, c.w,
                        cdiv.x, cdiv.y, cdiv.z
                      );    
            }
            else
            {
                printf("RendererBase::dump %6u/%6u : world %15f %15f %15f  (no composition) \n", i,count,
                        x, y, z
                      );    
 
            }
        }
    }
}

void Renderer::dump(const char* msg)
{
    printf("%s\n", msg );
    printf("vertices  %u \n", m_vbuf->getBufferId());
    printf("normals   %u \n", m_nbuf->getBufferId());
    printf("colors    %u \n", m_cbuf->getBufferId());
    printf("indices   %u \n", m_fbuf->getBufferId());
    printf("nelem     %d \n", m_indices_count);
    printf("hasTex    %d \n", hasTex());
    printf("shaderdir %s \n", getShaderDir());
    printf("shadertag %s \n", getShaderTag());

    //m_shader->dump(msg);
}

void Renderer::Print(const char* msg)
{
    printf("Renderer::%s tag %s nelem %d vao %d \n", msg, getShaderTag(), m_indices_count, m_vao[0] );
}

