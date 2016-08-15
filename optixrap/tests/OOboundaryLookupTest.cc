#include "NPY.hpp"

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

    blib->createDynamicBuffers();

    NPY<float>* ori = blib->getBuffer() ; 
    ori->save("$TMP/OOboundaryLookupTest/ori.npy");

    //bool use_debug_buffer = true ;  
    bool use_debug_buffer = false ; 

    NPY<float>* inp = use_debug_buffer ? NPY<float>::make_dbg_like(ori, 0) : ori ; 
    //inp->dump();
    inp->save("$TMP/OOboundaryLookupTest/inp.npy");


    OBndLib obnd(context, blib );
    if(use_debug_buffer)
    {
        LOG(warning) << "OOboundaryTest replacing real properties buffer with debug buffer, with an index" ; 
        obnd.setDebugBuffer(inp);
    }
    obnd.convert();     // places boundary_texture, boundary_domain  into OptiX context 


    unsigned int nx = obnd.getWidth();  // number of wavelength samples
    unsigned int ny = obnd.getHeight(); // number of float4 props
    NPY<float>* out = NPY<float>::make(nx, ny, 4);


    optix::Buffer outBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, nx, ny);
    context["out_buffer"]->setBuffer(outBuffer);   

    OContext::Mode_t mode = ok.isCompute() ? OContext::COMPUTE : OContext::INTEROP ;

    OContext* m_ocontext(NULL);
    m_ocontext = new OContext(context, mode);

    optix::Group top = m_ocontext->getTop();

    const char* builder = "NoAccel" ;
    const char* traverser = "NoAccel" ; 
    optix::Acceleration acceleration = context->createAcceleration(builder, traverser);
    top->setAcceleration(acceleration);

    unsigned eight = BOUNDARY_NUM_MATSUR*BOUNDARY_NUM_FLOAT4 ; 
    assert(ny % eight == 0 );

    OLaunchTest ott(m_ocontext, &ok, "boundaryLookupTest.cu.ptx", "boundaryLookupTest", "exception");
    ott.setWidth( nx);
    ott.setHeight(ny / 8);
    ott.launch();

    out->read( outBuffer->map() );
    outBuffer->unmap(); 

    //out->dump();
    out->save("$TMP/OOboundaryLookupTest/out.npy");

    //bool dump = true ;  
    bool dump = false ;  
    float maxdiff = inp->maxdiff(out, dump);
    LOG(info) << "maxdiff " << maxdiff  ;
    assert(maxdiff < 1e-6 ); 


    return 0 ;     
}

