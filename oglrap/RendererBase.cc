#include <cstring>
#include <GL/glew.h>

#include "RendererBase.hh"
#include "Prog.hh"

#include "PLOG.hh"

#include "OGLRAP_BODY.hh"




const char* RendererBase::getShaderTag() const 
{
    return m_shadertag ;
}
const char* RendererBase::getShaderDir() const 
{
    return m_shaderdir ;
}
const char* RendererBase::getInclPath() const 
{
    return m_incl_path ;
}





RendererBase::RendererBase(const char* tag, const char* dir, const char* incl_path)
    :
    m_shader(NULL),
    m_program(-1),
    m_shaderdir(dir ? strdup(dir) : getenv("SHADER_DIR")),
    m_shadertag(strdup(tag)),
    m_incl_path(incl_path ? strdup(incl_path) : getenv("SHADER_INCL_PATH"))
{
    // no OpenGL context needed, just reads sources
    m_shader = new Prog(m_shaderdir, m_shadertag, m_incl_path ); 
}


void RendererBase::make_shader()
{
    LOG(debug) << "RendererBase::make_shader " 
              << " shaderdir " << getShaderDir()
              << " shadertag " << getShaderTag()
              ;

    m_shader->createAndLink();
    //m_shader->Summary("RendererBase::make_shader");
    m_program = m_shader->getId(); 

    LOG(debug) << "RendererBase::make_shader "
              << " shaderdir " << getShaderDir()
              << " shadertag " << getShaderTag()
              << " program " <<  m_program ;

}


