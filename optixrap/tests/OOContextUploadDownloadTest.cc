#include "Opticks.hh"
#include "OpticksBufferControl.hh"

#include "OptiXTest.hh"
#include "OContext.hh"

#include "NPY.hpp"

#include "OPTICKS_LOG.hh"


int main( int argc, char** argv ) 
{
    OPTICKS_LOG(argc, argv);

    Opticks* ok = new Opticks(argc, argv, "--compute");
    ok->configure();


    OContext* ctx = OContext::Create(ok );
    optix::Context context = ctx->getContext(); 

    unsigned entry = ctx->addEntry("minimalTest.cu", "minimal", "exception");

    unsigned ni = 100 ; 
    unsigned nj = 4 ; 
    unsigned nk = 4 ; 

    NPY<float>* npy = NPY<float>::make(ni, nj, nk) ;
    npy->fill(42.);
    npy->save("$TMP/OOContextUploadDownloadTest_0.npy");
    npy->setBufferControl(OpticksBufferControl::Parse("OPTIX_SETSIZE,OPTIX_INPUT_OUTPUT"));

    optix::Buffer buffer = ctx->createBuffer<float>( npy, "demo");
    context["output_buffer"]->set(buffer);

    ctx->launch( OContext::VALIDATE,  entry, ni, 1);
    ctx->launch( OContext::COMPILE,   entry, ni, 1);
    ctx->launch( OContext::PRELAUNCH, entry, ni, 1);
    ctx->launch( OContext::LAUNCH,    entry, ni, 1);

    npy->zero();

    OContext::download( buffer, npy );

    NPYBase::setGlobalVerbose();

    npy->dump();
    npy->save("$TMP/OOContextUploadDownloadTest_1.npy");

    delete ctx ; 


    return 0;
}
