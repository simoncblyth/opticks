#include "Opticks.hh"
#include "OpticksBufferControl.hh"

#include "OptiXTest.hh"
#include "OContext.hh"

#include "NPY.hpp"

#include "OPTICKS_LOG.hh"


int main( int argc, char** argv ) 
{
    OPTICKS_LOG(argc, argv);

    Opticks ok(argc, argv, "--compute --printenabled");
    ok.configure();

    const char* cmake_target = "writeBufferTest" ;
    const char* ptxrel = "tests" ; 
    OContext* ctx = OContext::Create(&ok, cmake_target, ptxrel );
    optix::Context context = ctx->getContext(); 

    unsigned entry = ctx->addEntry("writeBufferTest.cu", "writeBuffer", "exception");

    unsigned ni = 100 ; 
    unsigned nj = 4 ; 
    unsigned nk = 4 ; 

    NPY<float>* npy = NPY<float>::make(ni, nj, nk) ;
    npy->zero();

    //const char* ctrl = "OPTIX_SETSIZE,OPTIX_INPUT_OUTPUT" ;  //  coming out zeros ??
    const char* ctrl = "OPTIX_OUTPUT_ONLY" ;
 
    npy->setBufferControl(OpticksBufferControl::Parse(ctrl));

    optix::Buffer buffer = ctx->createBuffer<float>( npy, "demo");

    context["output_buffer"]->set(buffer);

    ctx->launch( OContext::VALIDATE | OContext::COMPILE | OContext::PRELAUNCH | OContext::LAUNCH ,    entry, ni, 1);


    OContext::download( buffer, npy );

    NPYBase::setGlobalVerbose();

    npy->dump();
    npy->save("$TMP/writeBufferTest.npy");

    delete ctx ; 


    return 0;
}
