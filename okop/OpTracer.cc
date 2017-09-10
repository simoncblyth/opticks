#include "SLog.hh"
#include "SPPM.hh"
#include "PLOG.hh"
// npy-
#include "NGLM.hpp"
#include "NPY.hpp"

// okc-
#include "Composition.hh"

// okg-
#include "OpticksHub.hh"

// optixrap-
#include "OContext.hh"
#include "OTracer.hh"

#include "OXPPNS.hh"
#include <optixu/optixu_math_namespace.h>

// opop-
#include "OpEngine.hh"
#include "OpTracer.hh"


OpTracer::OpTracer(OpEngine* ope, OpticksHub* hub, bool immediate) 
   :
      m_log(new SLog("OpTracer::OpTracer")),
      m_ope(ope),
      m_hub(hub),
      m_immediate(immediate),

      m_ocontext(NULL),   // defer 
      m_composition(m_hub->getComposition()),
      m_otracer(NULL)
{
    init();
    (*m_log)("DONE");
}

void OpTracer::init()
{
    if(m_immediate)
    {
        prepareTracer();
    }
}


void OpTracer::prepareTracer()
{
    unsigned int width  = m_composition->getPixelWidth();
    unsigned int height = m_composition->getPixelHeight();


    LOG(debug) << "OpTracer::prepareTracer" 
               << " width " << width 
               << " height " << height 
                ;

    m_ocontext = m_ope->getOContext();

    optix::Context context = m_ocontext->getContext();

    optix::Buffer output_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, width, height);

    context["output_buffer"]->set( output_buffer );

    m_otracer = new OTracer(m_ocontext, m_composition);

}

void OpTracer::render()
{     
    m_hub->setupCompositionTargetting();

    m_otracer->setResolutionScale(1) ;
    m_otracer->trace_();
}   

void OpTracer::snap()
{
    LOG(info) << "OpTracer::snap START" ;
    render();
    LOG(info) << "OpTracer::snap DONE " ;


    m_ocontext->save("/tmp/snap.npy");
    m_ocontext->snap("/tmp/snap.ppm");
   

}
  




