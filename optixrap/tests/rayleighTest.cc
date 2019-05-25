#include "SSys.hh"
#include "BFile.hh"
#include "NPY.hpp"


#include "OLaunchTest.hh"
#include "OConfig.hh"
#include "OContext.hh"
#include "Opticks.hh"
#include "OpticksHub.hh"

#include "ORng.hh"
#include "OScene.hh"

#include "OPTICKS_LOG.hh"


/**
ORayleighTest
===================

**/


struct rayleighTest 
{
    rayleighTest( Opticks* ok, OContext* ocontext, optix::Context& context)
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


        // rayleighTest uses standard PTX (for loading geometry, materials etc..)
        // as well as test PTX. Hence need to shift the PTX sourcing. 
        OConfig* cfg = ocontext->getConfig();
        cfg->setCMakeTarget("rayleighTest");
        cfg->setPTXRel("tests");

        OLaunchTest ott(ocontext, ok, "rayleighTest.cu", "rayleighTest", "exception");
        ott.setWidth(  nx );   
        ott.setHeight( ny/4 );   // each thread writes 4*float4 
        ott.launch();

        NPY<float>* out = NPY<float>::make(nx, ny, 4);
        out->read( rayleighBuffer->map() );
        rayleighBuffer->unmap(); 
        out->save("$TMP/rayleighTest/ok.npy");
    }
};



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);    

    Opticks ok(argc, argv);
    OpticksHub hub(&ok);
    OScene sc(&hub);

    LOG(info) << " ok " ; 

    OContext* ocontext = sc.getOContext();
    ORng orng(&ok, ocontext);

    optix::Context context = ocontext->getContext();

    rayleighTest rt(&ok, ocontext, context); 

    return 0 ;
}

