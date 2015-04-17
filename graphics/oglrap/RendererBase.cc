#include <GL/glew.h>

#include "RendererBase.hh"
#include "Composition.hh"

#include "Prog.hh"

// npy-
#include "GLMPrint.hpp"

#include <glm/glm.hpp>  
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/type_ptr.hpp>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


RendererBase::RendererBase(const char* tag)
    :
    m_shader(NULL),
    m_shaderdir(NULL),
    m_shadertag(NULL),
    m_composition(NULL),
    m_program(-1)
{
    setShaderTag(tag);
}

void RendererBase::setComposition(Composition* composition)
{
    m_composition = composition ;
}
void RendererBase::setShaderDir(const char* dir)
{
    m_shaderdir = strdup(dir);
}
void RendererBase::setShaderTag(const char* tag)
{
    m_shadertag = strdup(tag);
}


Composition* RendererBase::getComposition()
{
    return m_composition ;
}
char* RendererBase::getShaderDir()
{
    return m_shaderdir ? m_shaderdir : getenv("SHADER_DIR") ;
}
char* RendererBase::getShaderTag()
{
    return m_shadertag ? m_shadertag : getenv("SHADER_TAG") ;
}


void RendererBase::make_shader()
{

    m_shader = new Prog(getShaderDir(), getShaderTag(), true); 
    m_shader->createAndLink();
    m_shader->Summary("RendererBase::make_shader");
    m_program = m_shader->getId(); 

    // transitional
    // hmm this needs to be done at higher level, as differnent subclasses have differing needs 
    m_mvp_location = m_shader->uniform("ModelViewProjection", false) ; 
    m_mv_location = m_shader->uniform("ModelView", false);      // not required

    LOG(info) << "RendererBase::make_shader "
              << " shaderdir " << getShaderDir()
              << " shadertag " << getShaderTag()
              << " program " <<  m_program
              << " mvp " << m_mvp_location
              << " mv " << m_mv_location ;
}


void RendererBase::update_uniforms()
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




void RendererBase::dump(void* data, unsigned int nbytes, unsigned int stride, unsigned long offset, unsigned int count )
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


