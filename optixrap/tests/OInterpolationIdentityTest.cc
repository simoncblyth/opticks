#include "SSys.hh"
#include "BFile.hh"

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


#include "SYSRAP_LOG.hh"
#include "OKCORE_LOG.hh"
#include "GGEO_LOG.hh"
#include "OXRAP_LOG.hh"

#include "PLOG.hh"

/**
OInterpolationIdentityTest
=============================


**/


void identityTest( Opticks* ok, OContext* ocontext, OBndLib* obnd, optix::Context& context)
{
    optix::uint4 args = optix::make_uint4(0,0,0,0); 
    context["boundary_test_args"]->setUint(args);

    unsigned nb = obnd->getNumBnd(); 
    unsigned nx = obnd->getWidth();  // number of wavelength samples
    unsigned ny = obnd->getHeight(); // total number of float4 props

    LOG(info) 
             << " identity_buffer "
             << " nb " << nb 
             << " nx " << nx 
             << " ny " << ny
             ; 

    optix::Buffer idBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, nx, ny);
    context["out_buffer"]->setBuffer(idBuffer);   

    unsigned eight = BOUNDARY_NUM_MATSUR*BOUNDARY_NUM_FLOAT4 ; 
    assert(ny % eight == 0 );

    OLaunchTest idtest(ocontext, ok, "OInterpolationTest.cu.ptx", "OIdentityTest", "exception");
    idtest.setWidth(  nx );   
    idtest.setHeight( ny/8 );   // kernel loops over eight for matsur and num_float4
    idtest.launch();

    NPY<float>* out = NPY<float>::make(nx, ny, 4);
    out->read( idBuffer->map() );
    idBuffer->unmap(); 
    out->save("$TMP/InterpolationTest/OInterpolationTest_identity.npy");
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
    optix::Context context = ocontext->getContext();

    OBndLib* obnd = sc.getOBndLib();

    identityTest( &ok, ocontext, obnd, context );

    std::string path = BFile::FormPath("$OPTICKS_HOME/optixrap/tests/OInterpolationTest_identity.py");
    return SSys::exec("python",path.c_str());
}

