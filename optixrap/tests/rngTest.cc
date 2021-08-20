#include "NPY.hpp"

#include "Opticks.hh"

#include "OContext.hh"
#include "ORng.hh"
#include "OLaunchTest.hh"

#include "OPTICKS_LOG.hh"

const char* TMPDIR = "$TMP/optixrap/rngTest" ; 

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);    

    Opticks ok(argc, argv, "--compute");
    ok.configure();

    const char* cmake_target = "rngTest" ; 
    const char* ptxrel = "tests" ; 
    OContext* ctx = OContext::Create(&ok, cmake_target, ptxrel);
    optix::Context context = ctx->getContext();

    ORng* orng ; 
    orng = new ORng(&ok, ctx); 
    assert( orng ); 
 
    unsigned nx = 1000000 ; 
    unsigned ny = 1 ; 

    optix::Buffer outBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, nx );
    context["out_buffer"]->setBuffer(outBuffer);   

    OLaunchTest ott(ctx, &ok, "rngTest.cu", "rngTest", "exception");
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

