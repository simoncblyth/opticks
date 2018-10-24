#include <cassert>
#include <sstream>
#include "PLOG.hh"

#include "OpticksConst.hh"
#include "Composition.hh"
#include "RenderStyle.hh"

const char* RenderStyle::R_PROJECTIVE_ = "R_PROJECTIVE" ;
const char* RenderStyle::R_RAYTRACED_  = "R_RAYTRACED" ; 
const char* RenderStyle::R_COMPOSITE_  = "R_COMPOSITE" ; 


RenderStyle::RenderStyle(Composition* composition) 
    :
    m_composition(composition),
    m_render_style(R_PROJECTIVE),
    m_raytrace_enabled(false)    // <-- enabled by OKGLTracer 
{
}


const char* RenderStyle::RenderStyleName(RenderStyle_t style) // static
{
    const char* s = NULL ; 
    switch(style)
    { 
       case R_PROJECTIVE: s = R_PROJECTIVE_ ; break ; 
       case R_RAYTRACED:  s = R_RAYTRACED_  ; break ; 
       case R_COMPOSITE:  s = R_COMPOSITE_  ; break ; 
       case NUM_RENDER_STYLE: s = NULL      ; break ; 
    }
    assert(s);  
    return s ; 
}


const char* RenderStyle::getRenderStyleName() const 
{
    return RenderStyleName(getRenderStyle()) ; 
}

RenderStyle::RenderStyle_t RenderStyle::getRenderStyle() const 
{
    return m_render_style ; 
}

bool RenderStyle::isProjectiveRender() const 
{
   return m_render_style == R_PROJECTIVE ;
}
bool RenderStyle::isRaytracedRender() const 
{
   return m_render_style == R_RAYTRACED ;
}
bool RenderStyle::isCompositeRender() const 
{
   return m_render_style == R_COMPOSITE ;
}


std::string RenderStyle::desc() const 
{
    std::stringstream ss ; 
    ss << "RenderStyle "
       << getRenderStyleName() 
       ;
    return ss.str(); 
}




void RenderStyle::setRaytraceEnabled(bool raytrace_enabled) // set by OKGLTracer
{
    m_raytrace_enabled = raytrace_enabled ;
}

void RenderStyle::nextRenderStyle(unsigned modifiers)  // O:key cycling: Projective, Raytraced, Composite 
{
    if(!m_raytrace_enabled)
    {
        LOG(error) << "RenderStyle::nextRenderStyle is inhibited as RenderStyle::setRaytraceEnabled has not been called, see okgl.OKGLTracer " ;  
        return ; 
    }


    bool nudge = modifiers & OpticksConst::e_shift ;
    if(nudge)
    {
        m_composition->setChanged(true) ;
        return ; 
    }

    int next = (m_render_style + 1) % NUM_RENDER_STYLE ; 

    m_render_style = (RenderStyle_t)next ; 

    applyRenderStyle();

    m_composition->setChanged(true) ; // trying to avoid the need for shift-O nudging 
}

void RenderStyle::applyRenderStyle()   
{
    // nothing to do, style is honoured by  Scene::render
}



