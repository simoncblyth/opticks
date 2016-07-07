#include "OTextureTest.hh"

// optickscore-
#include "Opticks.hh"

// optixrap-
#include "OContext.hh"

// optix-
#include <optixu/optixu.h>
#include <optixu/optixu_math_stream_namespace.h>

using namespace optix ; 

#include "PLOG.hh"

OTextureTest::OTextureTest(OContext* ocontext, Opticks* opticks) 
   :
    m_ocontext(ocontext),
    m_opticks(opticks),

    m_entry_index(-1),
    m_width(1),
    m_height(1)
{
    init();
}

void OTextureTest::init()
{
    m_context = m_ocontext->getContext();
    m_entry_index = m_ocontext->addRayGenerationProgram(   "textureTest.cu.ptx", "textureTest" );
    int exception_index = m_ocontext->addExceptionProgram( "textureTest.cu.ptx", "exception");
    assert(m_entry_index == exception_index);
}

void OTextureTest::launch()
{
    m_ocontext->launch( OContext::VALIDATE|OContext::COMPILE|OContext::PRELAUNCH,  m_entry_index ,  m_width, m_height);
    m_ocontext->launch( OContext::LAUNCH,  m_entry_index,  m_width, m_height);
}



