#include "SLog.hh"
#include "PLOG.hh"
// npy-
#include "NGLM.hpp"

// okc-
#include "Composition.hh"

// okg-
#include "OpticksHub.hh"

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
#include "OpticksViz.hh"

// opgl-
#include "OKGLTracer.hh"
#include "OFrame.hh"
#include "ORenderer.hh"


OKGLTracer::OKGLTracer(OpEngine* ope, OpticksViz* viz, bool immediate) 
   :
      m_log(new SLog("OKGLTracer::OKGLTracer")),
      m_ope(ope),
      m_viz(viz),
      m_hub(m_viz->getHub()),
      m_immediate(immediate),
      m_scene(m_viz->getScene()),

      m_ocontext(NULL),   // defer 
      m_composition(m_hub->getComposition()),
      m_interactor(m_viz->getInteractor()),
      m_oframe(NULL),
      m_orenderer(NULL),
      m_otracer(NULL),
      m_trace_count(0)
{
    init();
    (*m_log)("DONE");
}

void OKGLTracer::init()
{
    if(m_immediate)
    {
        prepareTracer();
    }
}


void OKGLTracer::prepareTracer()
{
    if(m_hub->isCompute()) return ;
    if(!m_scene) 
    {
        LOG(fatal) << "OKGLTracer::prepareTracer NULL scene ?"  ;
        return ;
    }


    m_viz->setExternalRenderer(this);

    unsigned int width  = m_composition->getPixelWidth();
    unsigned int height = m_composition->getPixelHeight();

    LOG(debug) << "OKGLTracer::prepareTracer plant external renderer into viz" 
               << " width " << width 
               << " height " << height 
                ;

    m_ocontext = m_ope->getOContext();

    optix::Context context = m_ocontext->getContext();

    m_oframe = new OFrame(context, width, height);

    context["output_buffer"]->set( m_oframe->getOutputBuffer() );

    m_interactor->setTouchable(m_oframe);

    Renderer* rtr = m_scene->getRaytraceRenderer();

    m_orenderer = new ORenderer(rtr, m_oframe, m_scene->getShaderDir(), m_scene->getShaderInclPath());

    m_otracer = new OTracer(m_ocontext, m_composition);

    //m_ocontext->dump("OKGLTracer::prepareTracer");
}

void OKGLTracer::render()
{     
    if(m_otracer && m_orenderer)
    { 
        if(m_composition->hasChangedGeometry())
        {
            unsigned int scale = m_interactor->getOptiXResolutionScale() ;
            m_otracer->setResolutionScale(scale) ;
            m_otracer->trace_();
            m_oframe->push_PBO_to_Texture();

/*
            if(m_trace_count == 0 )
            {
                LOG(info) << "OKGLTracer::render snapping first raytrace frame " ; 
                m_ocontext->snap();
            }
*/


            m_trace_count++ ; 
        }
        else
        {
            // dont bother tracing when no change in geometry
        }
    }
    
}   



