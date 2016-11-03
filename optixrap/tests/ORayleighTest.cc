#include "SSys.hh"
#include "BFile.hh"
#include "NPY.hpp"


#include "OLaunchTest.hh"
#include "OContext.hh"
#include "Opticks.hh"
#include "OpticksHub.hh"

#include "ORng.hh"
#include "OScene.hh"


#include "SYSRAP_LOG.hh"
#include "OKCORE_LOG.hh"
#include "GGEO_LOG.hh"
#include "OXRAP_LOG.hh"

#include "PLOG.hh"

/**
ORayleighTest
===================

**/

void rayleighTest( Opticks* ok, OContext* ocontext, optix::Context& context)
{
    unsigned nx = 1000000 ; 
    unsigned ny = 4 ;      // total number of float4 props

    LOG(info) 
             << " rayleigh_buffer "
             << " nx " << nx 
             << " ny " << ny
             ; 

    optix::Buffer rayleighBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, nx, ny);
    context["rayleigh_buffer"]->setBuffer(rayleighBuffer);   


    OLaunchTest ott(ocontext, ok, "ORayleighTest.cu.ptx", "ORayleighTest", "exception");
    ott.setWidth(  nx );   
    ott.setHeight( ny/4 );   // each thread writes 4*float4 
    ott.launch();

    NPY<float>* out = NPY<float>::make(nx, ny, 4);
    out->read( rayleighBuffer->map() );
    rayleighBuffer->unmap(); 
    out->save("$TMP/RayleighTest/ok.npy");
}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);    

    SYSRAP_LOG__ ; 
    OKCORE_LOG__ ; 
    GGEO_LOG__ ; 
    OXRAP_LOG__ ; 

    Opticks ok(argc, argv);
    OpticksHub hub(&ok);
    OScene sc(&hub);

    LOG(info) << " ok " ; 

    OContext* ocontext = sc.getOContext();
    ORng orng(&ok, ocontext);

    optix::Context context = ocontext->getContext();

    rayleighTest( &ok, ocontext, context) ;

    return 0 ;
}

