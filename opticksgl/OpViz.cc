// npy-
#include "NLog.hpp"

// opticks-
#include "Composition.hh"

// optixrap-
#include "OContext.hh"
#include "OTracer.hh"

// opop-
#include "OpEngine.hh"

// oglrap-  Frame brings in GL/glew.h GLFW/glfw3.h gleq.h
#include "Frame.hh"
#include "Scene.hh"
#include "Interactor.hh"
#include "Renderer.hh"
#include "Rdr.hh"

// opgl-
#include "OpViz.hh"
#include "OFrame.hh"
#include "ORenderer.hh"


void OpViz::init()
{
    m_opticks = m_ope->getOpticks();
    m_ocontext = m_ope->getOContext();

    if(m_opticks->isCompute()) return ;
    if(!m_scene) return ;

    m_composition = m_scene->getComposition();
    m_interactor = m_scene->getInteractor();

    prepareOptiXViz();
}



void OpViz::prepareOptiXViz()
{
    unsigned int width  = m_composition->getPixelWidth();
    unsigned int height = m_composition->getPixelHeight();

    optix::Context context = m_ocontext->getContext();

    m_oframe = new OFrame(context, width, height);

    context["output_buffer"]->set( m_oframe->getOutputBuffer() );

    m_interactor->setTouchable(m_oframe);

    Renderer* rtr = m_scene->getRaytraceRenderer();

    m_orenderer = new ORenderer(rtr, m_oframe, m_scene->getShaderDir(), m_scene->getShaderInclPath());

    m_otracer = new OTracer(m_ocontext, m_composition);

    LOG(info) << "OpViz::prepareOptiXViz DONE ";

    m_ocontext->dump("OpViz::prepareOptiXVix");
}


void OpViz::render()
{     
    if(m_otracer && m_orenderer)
    { 
        if(m_composition->hasChangedGeometry())
        {
            unsigned int scale = m_interactor->getOptiXResolutionScale() ;
            m_otracer->setResolutionScale(scale) ;
            m_otracer->trace();
            m_oframe->push_PBO_to_Texture();
        }
        else
        {
            // dont bother tracing when no change in geometry
        }
    }
}   




