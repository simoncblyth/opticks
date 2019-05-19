#include "SLog.hh"
#include "SSys.hh"
#include "SPPM.hh"
#include "BFile.hh"
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
               << " immediate " << m_immediate
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
at various positions configured via --snapconfig

**/

void OpTracer::snap()   // --snapconfig="steps=5,eyestartz=0,eyestopz=0"
{
    LOG(info) << "(" << m_snap_config->desc();

    int num_steps = m_snap_config->steps ; 

    float eyestartx = m_snap_config->eyestartx ; 
    float eyestarty = m_snap_config->eyestarty ; 
    float eyestartz = m_snap_config->eyestartz ; 

    float eyestopx = m_snap_config->eyestopx ; 
    float eyestopy = m_snap_config->eyestopy ; 
    float eyestopz = m_snap_config->eyestopz ; 

    for(int i=0 ; i < num_steps ; i++)
    {
        std::string path_ = m_snap_config->getSnapPath(i) ; 
        bool create = true ; 
        std::string path = BFile::preparePath(path_.c_str(), create);  


        float frac = num_steps > 1 ? float(i)/float(num_steps-1) : 0.f ; 

        float eyex = m_composition->getEyeX();
        float eyey = m_composition->getEyeY();
        float eyez = m_composition->getEyeZ();

        if(!SSys::IsNegativeZero(eyestartx))
        { 
            eyex = eyestartx + (eyestopx-eyestartx)*frac ; 
            m_composition->setEyeX( eyex ); 
        }
        if(!SSys::IsNegativeZero(eyestarty))
        { 
            eyey = eyestarty + (eyestopy-eyestarty)*frac ; 
            m_composition->setEyeY( eyey ); 
        }
        if(!SSys::IsNegativeZero(eyestartz))
        { 
            eyez = eyestartz + (eyestopz-eyestartz)*frac ; 
            m_composition->setEyeZ( eyez ); 
        }

        render();

        std::cout << " i " << std::setw(5) << i 
                  << " eyex " << std::setw(10) << eyex
                  << " eyey " << std::setw(10) << eyey
                  << " eyez " << std::setw(10) << eyez
                  << " path " << path 
                  << std::endl ;         

        m_ocontext->snap(path.c_str());
    }

    m_otracer->report("OpTracer::snap");   // saves for runresultsdir
    //m_ok->dumpMeta("OpTracer::snap");

    m_ok->saveParameters(); 

    LOG(info) << ")" ;
}
  

