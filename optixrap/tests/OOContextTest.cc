#include "Opticks.hh"

#include "OptiXTest.hh"
#include "OContext.hh"

#include "NPY.hpp"

#include "OXRAP_LOG.hh"
#include "PLOG.hh"


int main( int argc, char** argv ) 
{
    PLOG_(argc, argv);
    OXRAP_LOG__ ; 

    Opticks* m_opticks = new Opticks(argc, argv);
    m_opticks->configure();

    optix::Context context = optix::Context::create();

    //OContext::Mode_t mode = m_opticks->isCompute() ? OContext::COMPUTE : OContext::INTEROP ;
    //OContext::Mode_t mode = OContext::INTEROP ;
    OContext::Mode_t mode = OContext::COMPUTE ;

    OContext* m_ocontext = new OContext(context, mode, false );

    unsigned entry = m_ocontext->addEntry("minimalTest.cu.ptx", "minimal", "exception");

    unsigned ni = 100 ; 
    unsigned nj = 4 ; 
    unsigned nk = 4 ; 

    NPY<float>* npy = NPY<float>::make(ni, nj, nk) ;

    optix::Buffer buffer = m_ocontext->createBuffer<float>( npy, "demo", OContext::BUFOPT_SETSIZE | OContext::BUFOPT_INPUT_OUTPUT);

    context["output_buffer"]->set(buffer);

    m_ocontext->launch( OContext::VALIDATE,  entry, ni, 1);
    m_ocontext->launch( OContext::COMPILE,   entry, ni, 1);
    m_ocontext->launch( OContext::PRELAUNCH, entry, ni, 1);
    m_ocontext->launch( OContext::LAUNCH,    entry, ni, 1);

    npy->zero();

    OContext::download( buffer, npy );

    NPYBase::setGlobalVerbose();

    npy->dump();
    npy->save("$TMP/OOContextTest.npy");

    return 0;
}
