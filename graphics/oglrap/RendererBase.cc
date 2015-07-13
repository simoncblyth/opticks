#include <GL/glew.h>

#include "RendererBase.hh"
#include "Prog.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


RendererBase::RendererBase(const char* tag, const char* dir, const char* incl_path)
    :
    m_shader(NULL),
    m_shaderdir(dir ? strdup(dir) : getenv("SHADER_DIR")),
    m_incl_path(incl_path ? strdup(incl_path) : getenv("SHADER_INCL_PATH")),
    m_shadertag(strdup(tag)),
    m_program(-1)
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


