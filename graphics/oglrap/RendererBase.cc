#include <GL/glew.h>

#include "RendererBase.hh"

#include "Shader.hh"
#include "Composition.hh"
#include "Common.hh"


#include <glm/glm.hpp>  
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/type_ptr.hpp>


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
    m_shader = new Shader(getShaderDir(), getShaderTag());
    m_program = m_shader->getId();

    m_mvp_location = m_shader->getMVPLocation();
    m_mv_location = m_shader->getMVLocation();
}



void RendererBase::update_uniforms()
{
    if(m_composition)
    {
        m_composition->update() ;
        // could cache the ptrs they aint changing 
        glUniformMatrix4fv(m_mv_location, 1, GL_FALSE, glm::value_ptr(m_composition->getWorld2Eye()));
        glUniformMatrix4fv(m_mvp_location, 1, GL_FALSE, glm::value_ptr(m_composition->getWorld2Clip()));
    } 
    else
    { 
        glm::mat4 identity ; 
        glUniformMatrix4fv(m_mv_location, 1, GL_FALSE, glm::value_ptr(identity));
        glUniformMatrix4fv(m_mvp_location, 1, GL_FALSE, glm::value_ptr(identity));
    }
}


void RendererBase::use_shader()
{
    m_shader->use();
}





