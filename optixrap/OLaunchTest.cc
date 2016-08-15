#include "OLaunchTest.hh"

// optickscore-
#include "Opticks.hh"

// optixrap-
#include "OContext.hh"

// optix-
#include <optixu/optixu.h>
#include <optixu/optixu_math_stream_namespace.h>

using namespace optix ; 

#include "PLOG.hh"

OLaunchTest::OLaunchTest(OContext* ocontext, Opticks* opticks, const char* ptx, const char* prog, const char* exception) 
   :
    m_ocontext(ocontext),
    m_opticks(opticks),
    m_ptx(strdup(ptx)),
    m_prog(strdup(prog)),
    m_exception(strdup(exception)),

    m_entry_index(-1),
    m_width(1),
    m_height(1)
{
    init();
}

void OLaunchTest::setWidth(unsigned int width)
{
    m_width = width ; 
}
void OLaunchTest::setHeight(unsigned int height)
{
    m_height = height ; 
}


void OLaunchTest::init()
{
    m_context = m_ocontext->getContext();
    m_entry_index = m_ocontext->addEntry(m_ptx, m_prog, m_exception);
}

void OLaunchTest::launch()
{
    m_ocontext->launch( OContext::VALIDATE|OContext::COMPILE|OContext::PRELAUNCH,  m_entry_index ,  m_width, m_height);
    m_ocontext->launch( OContext::LAUNCH,  m_entry_index,  m_width, m_height);
}



