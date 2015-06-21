#include <GL/glew.h>

#include "Renderer.hh"
#include "Prog.hh"
#include "Composition.hh"

// npy-
#include "GLMPrint.hpp"


#include <glm/glm.hpp>  
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/type_ptr.hpp>

// ggeo
#include "GArray.hh"
#include "GBuffer.hh"
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

void Renderer::setDrawable(GDrawable* drawable, bool debug)
{
    m_drawable = drawable ;
    gl_upload_buffers(debug);
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


void Renderer::gl_upload_buffers(bool debug)
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

    glGenVertexArrays (1, &m_vao); // OSX: undefined without glew 
    glBindVertexArray (m_vao);     

    GBuffer* vbuf = m_drawable->getVerticesBuffer();
    GBuffer* nbuf = m_drawable->getNormalsBuffer();
    GBuffer* cbuf = m_drawable->getColorsBuffer();
    GBuffer* ibuf = m_drawable->getIndicesBuffer();
    GBuffer* tbuf = m_drawable->getTexcoordsBuffer();
    setHasTex(tbuf != NULL);

    if(debug)
    {
        dump( vbuf->getPointer(),vbuf->getNumBytes(),vbuf->getNumElements()*sizeof(float),0,vbuf->getNumItems() ); 
    }

    assert(vbuf->getNumBytes() == cbuf->getNumBytes());
    assert(nbuf->getNumBytes() == cbuf->getNumBytes());

    // hmm maybe give GBuffers names ? and collect these identifiers into map
    // for more flexibility when NULL buf  
 
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


    glEnable(GL_CLIP_DISTANCE0); 
 
    make_shader();

    check_uniforms();

    glUseProgram(m_program);
}


void Renderer::check_uniforms()
{
    char* tag = getShaderTag();
 
    if(strcmp(tag,"nrm")==0)
    {
        m_mvp_location = m_shader->uniform("ModelViewProjection", true); 
        m_mv_location =  m_shader->uniform("ModelView",           true);      
        m_clip_location = m_shader->uniform("ClipPlane",          true); 
        m_param_location = m_shader->uniform("Param",          true); 
        m_lightposition_location = m_shader->uniform("LightPosition",true); 
    } 
    else if(strcmp(tag,"tex")==0)
    {
        m_mv_location =  m_shader->uniform("ModelView",           true);    
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
              << " clip " << m_clip_location ;

}

void Renderer::update_uniforms()
{
    if(m_composition)
    {
        m_composition->update() ;
        glUniformMatrix4fv(m_mv_location, 1, GL_FALSE,  m_composition->getWorld2EyePtr());
        glUniformMatrix4fv(m_mvp_location, 1, GL_FALSE, m_composition->getWorld2ClipPtr());

        glm::vec4 par = m_composition->getParam();
        glUniform4f(m_param_location, par.x, par.y, par.z, par.w  );    

        glm::vec4 lp = m_composition->getLightPosition();
        glUniform4f(m_lightposition_location, lp.x, lp.y, lp.z, lp.w  );    


        glUniform4fv(m_clip_location, 1, m_composition->getClipPlanePtr() );


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




void Renderer::render()
{ 
    glUseProgram(m_program);

    update_uniforms();

    glBindVertexArray (m_vao);

    // https://www.opengl.org/archives/resources/faq/technical/transparency.htm
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
    glEnable (GL_BLEND);



    glDrawElements( GL_TRIANGLES, m_indices_count, GL_UNSIGNED_INT, NULL ) ; // indices_count would be 3 for a single triangle 

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

