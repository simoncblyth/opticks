#include "GBndLib.hh"
#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"

#include "OBndLib.hh"
#include "OLaunchTest.hh"
#include "OContext.hh"
#include "Opticks.hh"

#include "OKCORE_LOG.hh"
#include "GGEO_LOG.hh"
#include "OXRAP_LOG.hh"

#include "PLOG.hh"

int main(int argc, char** argv)
{

    PLOG_(argc, argv);    

    OKCORE_LOG__ ; 
    GGEO_LOG__ ; 
    OXRAP_LOG__ ; 

    Opticks ok(argc, argv);
    ok.configure();

    LOG(info) << " ok " ; 

    GBndLib* blib = GBndLib::load(&ok);

    LOG(info) << " loaded blib " ; 
    GMaterialLib* mlib = GMaterialLib::load(&ok);
    GSurfaceLib*  slib = GSurfaceLib::load(&ok);

    LOG(info) << " loaded all " 
              << " blib " << blib
              << " mlib " << mlib
              << " slib " << slib
              ;

    blib->setMaterialLib(mlib);
    blib->setSurfaceLib(slib);
    blib->dump();

    optix::Context context = optix::Context::create();


    unsigned args_x = argc > 1 ? atoi(argv[1]) : 13 ; 
    unsigned args_y = argc > 2 ? atoi(argv[2]) :  0 ; 
    unsigned args_z = argc > 3 ? atoi(argv[3]) : 42 ; 
    unsigned args_w = argc > 4 ? atoi(argv[4]) : 42 ; 

    optix::uint4 args = optix::make_uint4(args_x, args_y, args_z, args_w ); 

    context["boundary_test_args"]->setUint(args);


    OBndLib obnd(context, blib );
    obnd.convert();     // places boundary_texture, boundary_domain  into OptiX context 


    OContext::Mode_t mode = ok.isCompute() ? OContext::COMPUTE : OContext::INTEROP ;

    OContext* m_ocontext(NULL);
    m_ocontext = new OContext(context, mode);

    optix::Group top = m_ocontext->getTop();

    const char* builder = "NoAccel" ;
    const char* traverser = "NoAccel" ; 
    optix::Acceleration acceleration = context->createAcceleration(builder, traverser);
    top->setAcceleration(acceleration);

    OLaunchTest ott(m_ocontext, &ok, "boundaryTest.cu.ptx", "boundaryTest", "exception");
    ott.launch();

    LOG(info) << "DONE" ; 


    return 0 ;     
}

