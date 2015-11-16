#include "ORenderer.hh"

#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

// npy-
#include "timeutil.hpp"

// optixrap-
#include "OFrame.hh"

// oglrap-
#include "Renderer.hh"
#include "Texture.hh"



void ORenderer::init(const char* dir, const char* incl_path)
{
    unsigned int width  = m_frame->getWidth();
    unsigned int height = m_frame->getHeight();

    // TODO: reposition Renderer externally ...
    m_renderer = new Renderer("tex", dir, incl_path );
    m_texture = new Texture();   // QuadTexture would be better name
    m_texture->setSize(width, height);
    m_texture->create();

    m_texture_id = m_texture->getTextureId() ;

    LOG(debug) << "ORenderer::init size(" << width << "," << height << ")  texture_id " << m_texture_id ;
    m_renderer->upload(m_texture);
}


void ORenderer::setSize(unsigned int width, unsigned int height)
{
    m_frame->setSize( width, height) ;

    //m_width = width ;
    //m_height = height ;

    //m_composition->setSize(width, height);
    m_texture->setSize(width, height);
}



void ORenderer::render()
{
    LOG(debug) << "ORenderer::render " << m_render_count ; 

    double t0 = getRealTime();

    m_frame->push_PBO_to_Texture(m_texture_id);

    double t1 = getRealTime();

    m_renderer->render();

    double t2 = getRealTime();

    m_render_count += 1 ; 
    m_render_prep += t1 - t0 ; 
    m_render_time += t2 - t1 ; 

    glBindTexture(GL_TEXTURE_2D, 0 );  

    if(m_render_count % 10 == 0) report("ORenderer::render"); 
}



void ORenderer::report(const char* msg)
{
    if(m_render_count == 0) return ; 

    std::stringstream ss ; 
    ss 
          << " render_count    " << std::setw(10) << m_render_count  << std::endl 
          << " render_prep     " << std::setw(10) << m_render_prep  << " avg " << std::setw(10) << m_render_prep/m_render_count  << std::endl
          << " render_time     " << std::setw(10) << m_render_time  << " avg " << std::setw(10) << m_render_time/m_render_count  << std::endl
           ;

    LOG(debug) << msg << std::endl << ss.str() ; 

}


