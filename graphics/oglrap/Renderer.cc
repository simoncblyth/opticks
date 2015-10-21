#include <GL/glew.h>

#include "Renderer.hh"
#include "Prog.hh"
#include "Composition.hh"
#include "Texture.hh"

// npy-
#include "GLMPrint.hpp"
#include "NSlice.hpp"


#include <glm/glm.hpp>  
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/type_ptr.hpp>

// ggeo
#include "GArray.hh"
#include "GBuffer.hh"
#include "GMergedMesh.hh"
#include "GBBoxMesh.hh"
#include "GDrawable.hh"

#include "stdio.h"
#include "stdlib.h"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


const char* Renderer::PRINT = "print" ; 

Renderer::~Renderer()
{
}

void Renderer::configureI(const char* name, std::vector<int> values )
{
    if(values.empty()) return ; 
    if(strcmp(name, PRINT)==0) Print("Renderer::configureI");
}


GLuint Renderer::upload(GLenum target, GLenum usage, GBuffer* buffer, const char* name)
{
    GLuint buffer_id ; 
    int prior_id = buffer->getBufferId();
    if(prior_id == -1)
    {
        glGenBuffers(1, &buffer_id);
        glBindBuffer(target, buffer_id);

        glBufferData(target, buffer->getNumBytes(), buffer->getPointer(), usage);

        buffer->setBufferId(buffer_id); 
        buffer->setBufferTarget(target); 

        LOG(debug) << "Renderer::upload " << name  ; 
        //buffer->Summary(name);
    }
    else
    {
        buffer_id = prior_id ; 
        LOG(debug) << "Renderer::upload binding to prior buffer : " << buffer_id ; 
        glBindBuffer(target, buffer_id);
    }
    return buffer_id ; 
}

void Renderer::upload(GBBoxMesh* bboxmesh, bool debug)
{
    m_bboxmesh = bboxmesh ;
    assert( m_geometry == NULL && m_texture == NULL );  // exclusive 
    m_drawable = static_cast<GDrawable*>(m_bboxmesh);
    NSlice* slice = m_bboxmesh->getSlice();
    upload_buffers(slice);
}
void Renderer::upload(GMergedMesh* geometry, bool debug)
{
    m_geometry = geometry ;
    assert( m_texture == NULL && m_bboxmesh == NULL );  // exclusive 
    m_drawable = static_cast<GDrawable*>(m_geometry);
    NSlice* slice = m_geometry->getSlice();
    upload_buffers(slice);
}
void Renderer::upload(Texture* texture, bool debug)
{
    m_texture = texture ;
    assert( m_geometry == NULL && m_bboxmesh == NULL ); // exclusive
    m_drawable = static_cast<GDrawable*>(m_texture);
    NSlice* slice = NULL ; 
    upload_buffers(slice);
}



void Renderer::upload_buffers(NSlice* slice)
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
    // TODO: adopt the more flexible ViewNPY approach used for event data
    //
    bool debug = false ; 

    assert(m_drawable);

    glGenVertexArrays (1, &m_vao); // OSX: undefined without glew 
    glBindVertexArray (m_vao);     

    GBuffer* vbuf = m_drawable->getVerticesBuffer();
    GBuffer* nbuf = m_drawable->getNormalsBuffer();
    GBuffer* cbuf = m_drawable->getColorsBuffer();
    GBuffer* fbuf = m_drawable->getIndicesBuffer();

    printf("Renderer::upload_buffers vbuf %p nbuf %p cbuf %p fbuf %p \n", vbuf, nbuf, cbuf, fbuf );

    GBuffer* tbuf = m_drawable->getTexcoordsBuffer();
    setHasTex(tbuf != NULL);

    GBuffer* ibuf_orig = m_drawable->getITransformsBuffer();
    GBuffer* ibuf = ibuf_orig ;
    setHasTransforms(ibuf != NULL);
    if(slice)
    {
        LOG(warning) << "Renderer::upload_buffers slicing ibuf with " << slice->description() ;
        ibuf = ibuf_orig->make_slice(slice); 
    }

    if(debug)
    {
        dump( vbuf->getPointer(),vbuf->getNumBytes(),vbuf->getNumElements()*sizeof(float),0,vbuf->getNumItems() ); 
    }

    assert(vbuf->getNumBytes() == cbuf->getNumBytes());
    assert(nbuf->getNumBytes() == cbuf->getNumBytes());

 
    m_vertices  = upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW,  vbuf, "vertices");
    m_normals   = upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW,  nbuf, "normals" );
    m_colors    = upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW,  cbuf, "colors" );

    if(hasTex())
    {
        m_texcoords = upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW,  tbuf, "texcoords" );
    }

    if(m_instanced) assert(hasTransforms()) ;

    if(hasTransforms())
    {
        m_transforms = upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW,  ibuf, "transforms");
        m_itransform_count = ibuf->getNumItems() ;
        LOG(info) << "Renderer::upload_buffers uploading transforms : itransform_count " << m_itransform_count ;
    }
    else
    {
        LOG(info) << "Renderer::upload_buffers NO TRANSFORMS " ;
    }


    m_indices  = upload(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, fbuf, "indices");
    m_indices_count = fbuf->getNumItems(); // number of indices = faces*3 ?

    GLboolean normalized = GL_FALSE ; 
    GLsizei stride = 0 ;

    const GLvoid* offset = NULL ;
 
    // the vbuf and cbuf NumElements refer to the number of elements 
    // within the vertex and color items ie 3 in both cases

    // CAUTION enum values vPosition, vNormal, vColor, vTexcoord 
    //         are duplicating layout numbers in the nrm/vert.glsl  
    // THIS IS FRAGILE
    //

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

    if(hasTransforms())
    {
        LOG(info) << "Renderer::upload_buffers setup transform attributes " ;
        glBindBuffer (GL_ARRAY_BUFFER, m_transforms);

        long qsize = sizeof(GLfloat) * 4 ;
        GLsizei matrix_stride = qsize * 4 ;

        glVertexAttribPointer(vTransform + 0 , 4, GL_FLOAT, normalized, matrix_stride, (void*)0 );
        glVertexAttribPointer(vTransform + 1 , 4, GL_FLOAT, normalized, matrix_stride, (void*)(qsize));
        glVertexAttribPointer(vTransform + 2 , 4, GL_FLOAT, normalized, matrix_stride, (void*)(qsize*2));
        glVertexAttribPointer(vTransform + 3 , 4, GL_FLOAT, normalized, matrix_stride, (void*)(qsize*3));

        glEnableVertexAttribArray (vTransform + 0);   
        glEnableVertexAttribArray (vTransform + 1);   
        glEnableVertexAttribArray (vTransform + 2);   
        glEnableVertexAttribArray (vTransform + 3);   

        glVertexAttribDivisor(vTransform + 0, 1);  // dictates instanced geometry shifts between instances
        glVertexAttribDivisor(vTransform + 1, 1);
        glVertexAttribDivisor(vTransform + 2, 1);
        glVertexAttribDivisor(vTransform + 3, 1);
    } 

    glEnable(GL_CLIP_DISTANCE0); 
 
    make_shader();

    glUseProgram(m_program);  // moved prior to check uniforms following Rdr::upload

    LOG(debug) <<  "Renderer::upload_buffers after make_shader " ; 
    check_uniforms();
    LOG(debug) <<  "Renderer::upload_buffers after check_uniforms " ; 

}


void Renderer::check_uniforms()
{
    std::string tag = getShaderTag();

    bool required = false;

    bool nrm  = tag.compare("nrm") == 0 ; 
    bool nrmvec = tag.compare("nrmvec") == 0 ; 
    bool inrm = tag.compare("inrm") == 0 ; 
    bool tex = tag.compare("tex") == 0 ; 

    LOG(debug) << "Renderer::check_uniforms " 
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
        m_lightposition_location = m_shader->uniform("LightPosition",required); 

        m_colordomain_location = m_shader->uniform("ColorDomain", required );     
        m_colors_location = m_shader->uniform("Colors", required );     

        if(inrm)
        {
            m_itransform_location = m_shader->uniform("InstanceTransform",required); 
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
    } 
    else
    {
        LOG(fatal) << "Renderer::checkUniforms unexpected shader tag " << tag ; 
        assert(0); 
    }

    LOG(debug) << "Renderer::check_uniforms "
              << " tag " << tag 
              << " mvp " << m_mvp_location
              << " mv " << m_mv_location 
              << " nrmparam " << m_nrmparam_location 
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

        glm::ivec4 np = m_composition->getNrmParam(); 
        glUniform4i(m_nrmparam_location, np.x, np.y, np.z, np.w);

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
        glm::mat4 identity ; 
        glUniformMatrix4fv(m_mv_location, 1, GL_FALSE, glm::value_ptr(identity));
        glUniformMatrix4fv(m_mvp_location, 1, GL_FALSE, glm::value_ptr(identity));
    }
}


void Renderer::bind()
{
    glBindVertexArray (m_vao);
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
    }
    // indices_count would be 3 for a single triangle, 30 for ten triangles


    //
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





void Renderer::dump(void* data, unsigned int nbytes, unsigned int stride, unsigned long offset, unsigned int count )
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
    printf("vertices  %u \n", m_vertices);
    printf("normals   %u \n", m_normals);
    printf("colors    %u \n", m_colors);
    printf("indices   %u \n", m_indices);
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

