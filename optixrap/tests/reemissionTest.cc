#include "NPY.hpp"

#include "Opticks.hh"
#include "GScintillatorLib.hh"

#include "OContext.hh"
#include "ORng.hh"
#include "OScintillatorLib.hh"
#include "OLaunchTest.hh"

#include "OPTICKS_LOG.hh"

const char* TMPDIR = "$TMP/optixrap/reemissionTest" ; 

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);    

    Opticks ok(argc, argv, "--compute");
    ok.configure();

    GScintillatorLib* slib = GScintillatorLib::load(&ok);
    slib->dump();

    const char* cmake_target = "reemissionTest" ; 
    const char* ptxrel = "tests" ; 
    OContext* ctx = OContext::Create(&ok, cmake_target, ptxrel);
    optix::Context context = ctx->getContext();

    ORng* orng ; 
    orng = new ORng(&ok, ctx); 
 
    OScintillatorLib* oscin ;  
    oscin = new OScintillatorLib(context, slib );

    const char* slice = "0:1" ; 
    oscin->convert(slice);

    LOG(info) << "DONE"  ;

    unsigned nx = 10000 ; 
    unsigned ny = 1 ; 

    optix::Buffer outBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, nx );
    context["out_buffer"]->setBuffer(outBuffer);   

    OLaunchTest ott(ctx, &ok, "reemissionTest.cu", "reemissionTest", "exception");
    ott.setWidth( nx);
    ott.setHeight(ny);
    ott.launch();


    NPY<float>* out = NPY<float>::make(nx);
    out->read( outBuffer->map() );
    outBuffer->unmap(); 

    LOG(info) << " save to " << TMPDIR << "/" << "out.npy" ; 
    out->save(TMPDIR, "out.npy");

    return 0 ;     
}

