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


OKGLTracer* OKGLTracer::fInstance = NULL ; 
OKGLTracer* OKGLTracer::GetInstance(){ return fInstance ;}

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
    fInstance = this ; 
}

void OKGLTracer::init()
{
    if(m_immediate)
    {
        prepareTracer();
    }
}



/**
OKGLTracer::prepareTracer
---------------------------

Establishes connection between: 

1. oxrap.OTracer m_otracer (OptiX) resident here
2. oglrap.Scene OpenGL "raytrace" renderer (actually its just renders tex pushed to it)

**/

void OKGLTracer::prepareTracer()
{
    if(m_hub->isCompute()) return ;
    if(!m_scene) 
    {
        LOG(fatal) << "OKGLTracer::prepareTracer NULL scene ?"  ;
        return ;
    }

    Scene* scene = Scene::GetInstance();
    assert(scene); 
 
    //scene->setRaytraceEnabled(true);  // enables the "O" key to switch to ray trace
    m_composition->setRaytraceEnabled(true);  // enables the "O" key to switch to ray trace


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


/**
OKGLTracer::render
--------------------

1. Invokes the OTracer::trace

2. Pushes the OptiX raytrace PBO to the OpenGL tex renderer, 
   using OpenGL interop to keep this all GPU side. 


This method is invoked from down in OGLRAP.OpticksViz by
virtue of the SRenderer protocol base, and the 
setting of this instance as the external renderer 
of OpticksViz.

OpticksViz::render coordinates the dance of the 
two renderers and GPU texture passing with::

    425 void OpticksViz::render()
    426 {
    427     m_frame->viewport();
    428     m_frame->clear();
    429 
    430     if(m_scene->isRaytracedRender() || m_scene->isCompositeRender())
    431     {
    432         if(m_external_renderer) m_external_renderer->render();
    433     }
    434 
    435     m_scene->render();
    436 }

**/


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


