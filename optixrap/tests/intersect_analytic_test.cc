/**
::

    intersect_analytic_test --cu intersect_analytic_dummy_test.cu
    intersect_analytic_test --cu intersect_analytic_torus_test.cu            
    intersect_analytic_test --cu intersect_analytic_sphere_test.cu
    intersect_analytic_test --cu intersect_analytic_cone_test.cu
    intersect_analytic_test --cu intersect_analytic_convexpolyhedron_test.cu

**/
#include "OptiXTest.hh"

#include "SPath.hh"
#include "OGeo.hh"
#include "NPY.hpp"
#include "OPTICKS_LOG.hh"


int main( int argc, char** argv ) 
{
    OPTICKS_LOG(argc, argv);
    const SAr& args = PLOG::instance->args ; 
    args.dump(); 

    const char* cu_name = args.get_arg_after("--cu", "intersect_analytic_torus_test.cu" ); 
    const char* progname = SPath::Stem(cu_name) ;        

    LOG(info) 
         << " cu_name " << cu_name 
         << " progname " << progname 
         ;

    optix::Context context = optix::Context::create();

    RTsize stack_size = context->getStackSize(); 
    LOG(info) << " stack_size " << stack_size ; 
    //context->setStackSize(6000);

    OptiXTest* test = new OptiXTest(context, cu_name, progname, "exception", "optixrap", "OptiXRap" ) ;

    std::cout << test->description() << std::endl ; 

    unsigned width = 1 ; 
    unsigned height = 1 ; 

    // optix::Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width, height );
    optix::Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width*height );
    context["output_buffer"]->set(buffer);


    NPY<float>* planBuf = NPY<float>::make(6, 4) ;  
    planBuf->zero();
    float hsize = 200.f ;
    unsigned j = 0 ; 
 
    planBuf->setQuad(0,j,  1.f, 0.f, 0.f,hsize );
    planBuf->setQuad(1,j, -1.f, 0.f, 0.f,hsize );
    planBuf->setQuad(2,j,  0.f, 1.f, 0.f,hsize );
    planBuf->setQuad(3,j,  0.f,-1.f, 0.f,hsize );
    planBuf->setQuad(4,j,  0.f, 0.f, 1.f,hsize );
    planBuf->setQuad(5,j,  0.f, 0.f,-1.f,hsize );

    unsigned verbosity = 3 ; 

    const char* ctxname = progname ;  // just informational

    optix::Buffer planBuffer = OGeo::CreateInputUserBuffer<float>( context, planBuf,  4*4, "planBuffer", ctxname, verbosity); 
    context["planBuffer"]->setBuffer(planBuffer);

    context->validate();
    context->compile();
    context->launch(0, width, height);


    NPY<float>* npy = NPY<float>::make(width, height, 4) ;
    npy->zero();

    void* ptr = buffer->map() ; 
    npy->read( ptr );
    buffer->unmap(); 

    const char* path = "$TMP/oxrap/intersect_analytic_test.npy";
    std::cerr << "save result npy to " << path << std::endl ; 
 
    npy->save(path);



    return 0;
}
