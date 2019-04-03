#include "SLog.hh"
#include "SPPM.hh"
#include "PLOG.hh"
// npy-
#include "NGLM.hpp"
#include "NPY.hpp"
#include "NSnapConfig.hpp"

// okc-
#include "Opticks.hh"
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

const plog::Severity OpTracer::LEVEL = debug ; 

OpTracer::OpTracer(OpEngine* ope, OpticksHub* hub, bool immediate) 
    :
    m_log(new SLog("OpTracer::OpTracer","",LEVEL)),
    m_ope(ope),
    m_hub(hub),
    m_ok(hub->getOpticks()),
    m_snap_config(m_ok->getSnapConfig()),
    m_immediate(immediate),

    m_ocontext(NULL),   // defer 
    m_composition(m_hub->getComposition()),
    m_otracer(NULL),
    m_count(0)
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
    if(m_count == 0 )
    {
        m_hub->setupCompositionTargetting();
        m_otracer->setResolutionScale(1) ;
    }

    m_otracer->trace_();
    m_count++ ; 
}   


/**
OpTracer::snap
----------------

Takes one or more GPU raytrace snapshots of geometry
at various positions configured via m_snap_config.  

**/

void OpTracer::snap()
{
    LOG(info) << "OpTracer::snap START" ;
    m_snap_config->dump();

    int num_steps = m_snap_config->steps ; 
    float eyestartz = m_snap_config->eyestartz ; 
    float eyestopz = m_snap_config->eyestopz ; 

    for(int i=0 ; i < num_steps ; i++)
    {
        std::string path = m_snap_config->getSnapPath(i) ; 

        float frac = num_steps > 1 ? float(i)/float(num_steps-1) : 0.f ; 
        float eyez = eyestartz + (eyestopz-eyestartz)*frac ; 

        std::cout << " i " << std::setw(5) << i 
                  << " eyez " << std::setw(10) << eyez
                  << " path " << path 
                  << std::endl ;         
   
        m_composition->setEyeZ( eyez ); 

        render();

        m_ocontext->snap(path.c_str());
    }
   
    LOG(info) << "OpTracer::snap DONE " ;
}
  

