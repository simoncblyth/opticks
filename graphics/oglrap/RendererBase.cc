#include <GL/glew.h>

#include "RendererBase.hh"

#include "Shader.hh"
#include "Composition.hh"
#include "Common.hh"


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
}

void RendererBase::use_shader()
{
    m_shader->use();
}





