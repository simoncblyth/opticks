#include <cstdint>

// brap-
#include "BBufSpec.hh"

// npy-
#include "NGLM.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"
#include "NSlice.hpp"



#include <GL/glew.h>



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
};



const char* Renderer::PRINT = "print" ; 


Renderer::Renderer(const char* tag, const char* dir, const char* incl_path)
    :
    RendererBase(tag, dir, incl_path),
    m_draw_num(0),
    m_draw_0(0),
    m_draw_1(1),

    m_vbuf(NULL),
    m_nbuf(NULL),
    m_cbuf(NULL),
    m_tbuf(NULL),
    m_fbuf(NULL),
    m_ibuf(NULL),

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
    m_instlodcull(NULL)
{
}


Renderer::~Renderer()
{
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


void Renderer::upload(GBBoxMesh* bboxmesh, bool /*debug*/)
{
    m_bboxmesh = bboxmesh ;
    assert( m_geometry == NULL && m_texture == NULL );  // exclusive 
    setDrawable(bboxmesh);
    upload();
    setupDraws(NULL) ; 
}
void Renderer::upload(GMergedMesh* mm, bool /*debug*/)
{
    m_geometry = mm ;
    assert( m_texture == NULL && m_bboxmesh == NULL );  // exclusive 
    setDrawable(mm);
    upload();
    setupDraws(mm);
}
void Renderer::upload(Texture* texture, bool /*debug*/)
{
    assert( m_geometry == NULL && m_bboxmesh == NULL );  // exclusive 
    setTexture(texture);
    upload();
    setupDraws(NULL) ; 
}
void Renderer::setTexture(Texture* texture)
{
    m_texture = texture ;
    m_texture_id = texture->getId();
    assert( m_geometry == NULL && m_bboxmesh == NULL ); // exclusive
    setDrawable(texture);
}
Texture* Renderer::getTexture() const 
{
    return m_texture ;
}

void Renderer::setDrawable(GDrawable* drawable) // CPU side buffer setup
{
    assert(drawable);
    m_drawable = drawable ;

    NSlice* islice = drawable->getInstanceSlice();
    NSlice* fslice = drawable->getFaceSlice();
 
    //  nvert: vertices, normals, colors
    m_vbuf = m_drawable->getVerticesBuffer();
    m_nbuf = m_drawable->getNormalsBuffer();
    m_cbuf = m_drawable->getColorsBuffer();

    assert(m_vbuf->getNumBytes() == m_cbuf->getNumBytes());
    assert(m_nbuf->getNumBytes() == m_cbuf->getNumBytes());
 
    // 3*nface indices
    GBuffer* fbuf_orig = m_drawable->getIndicesBuffer();
    m_fbuf = fslice ? fslice_element_buffer(fbuf_orig, fslice) : fbuf_orig ; 
    
    m_tbuf = m_drawable->getTexcoordsBuffer();
    setHasTex(m_tbuf != NULL);

    NPY<float>* ibuf = m_drawable->getITransformsBuffer();
    setHasTransforms(ibuf != NULL);

    if(islice)
        LOG(warning) << "Renderer::setDrawable instance slicing ibuf with " << islice->description() ;

    m_ibuf = islice ? ibuf->make_slice(islice) :  ibuf ; 

    bool debug = false ; 
    if(debug)
    {
        dump( m_vbuf->getPointer(),m_vbuf->getNumBytes(),m_vbuf->getNumElements()*sizeof(float),0,m_vbuf->getNumItems() ); 
    }

    if(m_instanced) assert(hasTransforms()) ;
}




GLuint Renderer::upload(GLenum target, GLenum usage, BBufSpec* spec, const char* name)
{
    GLuint buffer_id ; 
    int prior_id = spec->id ;

    if(prior_id == -1)
    {
        glGenBuffers(1, &buffer_id);
        glBindBuffer(target, buffer_id);

        glBufferData(target, spec->num_bytes, spec->ptr , usage);

        spec->id = buffer_id ; 
        spec->target = target ; 

        LOG(info) << "Renderer::upload " << std::setw(20) << name << " id " << buffer_id << " (FIRST) "  ; 
        //buffer->Summary(name);
    }
    else
    {
        buffer_id = prior_id ; 
        LOG(info) << "Renderer::upload " << std::setw(20) << name << " id " << buffer_id << " (BIND TO PRIOR) " ; 
        glBindBuffer(target, buffer_id);
    }
    return buffer_id ; 
}


void Renderer::upload_GBuffer(GLenum target, GLenum usage, GBuffer* buf, const char* name)
{
    if(!buf) return   ; 

    BBufSpec* spec = buf->getBufSpec(); 

    GLuint id = upload(target, usage, spec, name );

    buf->setBufferId(id);
    buf->setBufferTarget(target);

    LOG(trace) << "Renderer::upload_GBuffer" 
              << std::setw(20) << name 
              << " id " << std::setw(4) << id
              << " bytes " << std::setw(10) << spec->num_bytes
              ; 

}

void Renderer::upload_NPY(GLenum target, GLenum usage, NPY<float>* buf, const char* name)
{
    if(!buf) return   ; 
    BBufSpec* spec = buf->getBufSpec(); 

    GLuint id = upload(target, usage, spec, name );
    buf->setBufferId(id);


    buf->setBufferTarget(target);

    LOG(trace) << "Renderer::upload_NPY    " 
              << std::setw(20) << name 
              << " id " << std::setw(4) << id
              << " bytes " << std::setw(10) << spec->num_bytes
              ; 

}



void Renderer::upload()
{
    m_vao = createVertexArray(m_ibuf);

    make_shader();  // requires VAO bound to pass validation

    glUseProgram(m_program);  // moved prior to check uniforms following Rdr::upload

    glEnable(GL_CLIP_DISTANCE0); 
 
    check_uniforms();
}


GLuint Renderer::createVertexArray(NPY<float>* instanceBuffer)
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
    
     As there is only one GL_ELEMENT_ARRAY_BUFFER there is 
     no need to repeat the bind, but doing so for clarity
     
     TODO: consider adopting the more flexible ViewNPY approach used for event data
    
    */

    GLuint vao ; 
    glGenVertexArrays (1, &vao); 
    glBindVertexArray (vao);     

    // NB already uploaded buffers are just bind not uploaded again
    upload_GBuffer(GL_ARRAY_BUFFER, GL_STATIC_DRAW,  m_vbuf, "vertices");
    upload_GBuffer(GL_ARRAY_BUFFER, GL_STATIC_DRAW,  m_cbuf, "colors" );
    upload_GBuffer(GL_ARRAY_BUFFER, GL_STATIC_DRAW,  m_nbuf, "normals" );
    upload_GBuffer(GL_ARRAY_BUFFER, GL_STATIC_DRAW,  m_tbuf, "texcoords" );
    upload_GBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, m_fbuf, "indices");
    upload_NPY(GL_ARRAY_BUFFER, GL_STATIC_DRAW,  instanceBuffer , "transforms");

    m_itransform_count = instanceBuffer ? instanceBuffer->getNumItems() : 0 ;
    m_indices_count = m_fbuf->getNumItems(); // number of indices, would be 3 for a single triangle

    LOG(trace) << "Renderer::upload_buffers uploading transforms : itransform_count " << m_itransform_count ;


    GLuint instanceBO = instanceBuffer ? instanceBuffer->getBufferId() : 0u ;


    GLboolean normalized = GL_FALSE ; 
    GLsizei stride = 0 ;
    const GLvoid* offset = NULL ;
 
    glBindBuffer (GL_ARRAY_BUFFER, m_vbuf->getBufferId() );
    glVertexAttribPointer(vPosition, m_vbuf->getNumElements(), GL_FLOAT, normalized, stride, offset);
    glEnableVertexAttribArray (vPosition);  

    glBindBuffer (GL_ARRAY_BUFFER, m_nbuf->getBufferId() );
    glVertexAttribPointer(vNormal, m_nbuf->getNumElements(), GL_FLOAT, normalized, stride, offset);
    glEnableVertexAttribArray (vNormal);  

    glBindBuffer (GL_ARRAY_BUFFER, m_cbuf->getBufferId() );
    glVertexAttribPointer(vColor, m_cbuf->getNumElements(), GL_FLOAT, normalized, stride, offset);
    glEnableVertexAttribArray (vColor);   

    if(hasTex())
    {
        glBindBuffer (GL_ARRAY_BUFFER, m_tbuf->getBufferId()  );
        glVertexAttribPointer(vTexcoord, m_tbuf->getNumElements(), GL_FLOAT, normalized, stride, offset);
        glEnableVertexAttribArray (vTexcoord);   
    }

    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, m_fbuf->getBufferId() );

    if(hasTransforms())
    {
        LOG(trace) << "Renderer::upload_buffers setup instance transform attributes " ;
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
    return vao ; 
}









void Renderer::setupDraws(GMergedMesh* mm)
{
    // note distinction between: 
    //   1. a non-LOD draw 
    //   2. LOD set of one level 
    
    int num_comp = mm ? mm->getNumComponents() : 0  ; 

    if(mm == NULL || num_comp < 1 )
    {
        m_draw[0] = new DrawElements( GL_TRIANGLES, m_indices_count, GL_UNSIGNED_INT, NULL, m_itransform_count );
        m_draw_num = 1 ;  
        m_draw_0 = 0 ; 
        m_draw_1 = 1 ; 

        // indices_count would be 3 for a single triangle, 30 for ten triangles
    }
    else
    {
        LOG(info) << "Renderer::setupDraws"
                  << " m_indices_count " << m_indices_count
                   ;

        unsigned lod = 0 ; // full detail 
        //unsigned lod = 1 ; // bbox standin
        //unsigned lod = 2 ;   // quad standin

        mm->dumpComponents("Renderer::upload"); 

        assert( num_comp > 0 ) ;

        m_draw_num = num_comp ; 
        m_draw_0 = lod ; 
        m_draw_1 = m_draw_0 + 1 ; 

        for(unsigned i=0 ; i < m_draw_num ; i++)
        {
            glm::uvec4 eidx ;
            mm->getComponent(eidx, i );

            unsigned offset_face = eidx.x ;
            unsigned num_face = eidx.y ;
            GLsizei count = num_face*3 ; 
            void* offset_indices = (void*)(offset_face*3*sizeof(unsigned)) ; 

            m_draw[i] = new DrawElements( GL_TRIANGLES, count, GL_UNSIGNED_INT, offset_indices, 0 );
        }
    }
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

    LOG(trace) << "Renderer::check_uniforms " 
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

    LOG(trace) << "Renderer::check_uniforms "
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
        m_composition->update() ;
        glUniformMatrix4fv(m_mv_location, 1, GL_FALSE,  m_composition->getWorld2EyePtr());
        glUniformMatrix4fv(m_mvp_location, 1, GL_FALSE, m_composition->getWorld2ClipPtr());


        glUniform4fv(m_param_location, 1, m_composition->getParamPtr());

        glUniform4fv(m_scanparam_location, 1, m_composition->getScanParamPtr());
        glm::vec4 sp = m_composition->getScanParam(); 

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

        if(m_draw_count == 0)
            print( m_composition->getClipPlanePtr(), "Renderer::update_uniforms ClipPlane", 4);

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


void Renderer::bind()
{
    glBindVertexArray (m_vao);

    glActiveTexture(GL_TEXTURE0 + TEX_UNIT_0 );
    glBindTexture(GL_TEXTURE_2D,  m_texture_id );
}




void Renderer::render()
{ 
    glUseProgram(m_program);

    update_uniforms();

    bind();

    // https://www.opengl.org/archives/resources/faq/technical/transparency.htm
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
    glEnable (GL_BLEND);

    if(m_wireframe)
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    }



#ifdef OLD_DRAW
    if(m_instanced)
    {
        // primcount : Specifies the number of instances of the specified range of indices to be rendered.
        //             ie repeat sending the same set of vertices down the pipeline
        //
        GLsizei primcount = m_itransform_count ;  
        glDrawElementsInstanced( GL_TRIANGLES, m_indices_count, GL_UNSIGNED_INT, NULL, primcount  ) ;
    }
    else
    {

        glDrawElements( GL_TRIANGLES, m_indices_count, GL_UNSIGNED_INT, NULL ) ; 
       // indices_count would be 3 for a single triangle, 30 for ten triangles
    }
#else

    // hmm drawing multiple LOD levels only needed with dynamic LOD 
    // normally stick to single level

    for(unsigned i=m_draw_0 ; i < m_draw_1 ; i++)
    {
        const DrawElements& draw = *m_draw[i] ;   
        if(m_instanced)
        { 
            glDrawElementsInstanced( draw.mode, draw.count, draw.type,  draw.indices, draw.primcount  ) ;
        }
        else
        {
            glDrawElements( draw.mode, draw.count, draw.type,  draw.indices ) ;
        }
    }

#endif

    // TODO: try offsetting into the indices buffer using : (void*)(offset * sizeof(GLuint))
    //       eg to allow wireframing for selected volumes
    //
    //       need number of faces for every volume, so can cumsum*3 to get the indice offsets and counts 
    //
    //       http://stackoverflow.com/questions/9431923/using-an-offset-with-vbos-in-opengl
    //


    if(m_wireframe)
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }


    m_draw_count += 1 ; 

    glBindVertexArray(0);

    glUseProgram(0);
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
    printf("Renderer::%s tag %s nelem %d vao %d \n", msg, getShaderTag(), m_indices_count, m_vao );
}

