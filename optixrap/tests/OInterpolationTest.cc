#include "NPY.hpp"

#include "GBndLib.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"

#include "OBndLib.hh"
#include "OLaunchTest.hh"
#include "OContext.hh"
#include "Opticks.hh"
#include "OpticksHub.hh"

#include "OScene.hh"


#include "OKCORE_LOG.hh"
#include "GGEO_LOG.hh"
#include "OXRAP_LOG.hh"

#include "PLOG.hh"

/**
OInterpolationTest
===================


**/


int main(int argc, char** argv)
{

    PLOG_(argc, argv);    

    OKCORE_LOG__ ; 
    GGEO_LOG__ ; 
    OXRAP_LOG__ ; 

    Opticks ok(argc, argv);
    OpticksHub hub(&ok);
    OScene sc(&hub);

    LOG(info) << " ok " ; 

    OContext* ocontext = sc.getOContext();
    optix::Context context = ocontext->getContext();

    optix::uint4 args = optix::make_uint4(0,0,0,0); 
    context["boundary_test_args"]->setUint(args);

    OBndLib* obnd = sc.getOBndLib();
    unsigned nb = obnd->getNumBnd(); 
    unsigned nx = obnd->getWidth();  // number of wavelength samples
    unsigned ny = obnd->getHeight(); // total number of float4 props

    LOG(info) 
             << " nb " << nb 
             << " nx " << nx 
             << " ny " << ny
             << " ny % 8 " << ny % 8
             << " ny / 8 " << ny / 8
             ; 


    optix::Buffer outBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, nx, ny);
    context["out_buffer"]->setBuffer(outBuffer);   

    unsigned eight = BOUNDARY_NUM_MATSUR*BOUNDARY_NUM_FLOAT4 ; 
    assert(ny % eight == 0 );

    OLaunchTest ott(ocontext, &ok, "OInterpolationTest.cu.ptx", "OInterpolationTest", "exception");
    ott.setWidth(  nx );   
    ott.setHeight( ny/8 );   // kernel loops over eight for matsur and num_float4
    ott.launch();


    NPY<float>* out = NPY<float>::make(nx, ny, 4);
    out->read( outBuffer->map() );
    outBuffer->unmap(); 

    //out->dump();
    out->save("$TMP/InterpolationTest/OInterpolationTest.npy");


    return 0 ;     
}

