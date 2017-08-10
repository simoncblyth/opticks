
#include "OptiXMinimalTest.hh"


/*
#include "OGeo.hh"
#include "NPY.hpp"
#include "OXRAP_LOG.hh"
#include "PLOG.hh"
*/

int main( int argc, char** argv ) 
{
    //PLOG_(argc, argv);
    //OXRAP_LOG__ ; 

    optix::Context context = optix::Context::create();

    //OptiXTest* test = new OptiXTest(context, "intersect_analytic_test.cu", "intersect_analytic_test") ;

    assert( argc == 1 || argc == 3 );

   
    const char* ptxpath = argc == 1 ? "/usr/local/opticks/installcache/PTX/OptiXRap_generated_intersect_analytic_test.cu.ptx" : argv[1] ;
    const char* raygen = argc == 1 ? "intersect_analytic_test" : argv[2] ;

    std::cout 
       << " ptxpath " << ptxpath 
       << " raygen " << raygen
       << std::endl 
       ;


    OptiXMinimalTest* test = new OptiXMinimalTest(context, ptxpath, raygen ) ;

    std::cout << test->description() << std::endl ; 


    unsigned width = 1 ; 
    unsigned height = 1 ; 

    // optix::Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width, height );
    optix::Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width*height );
    context["output_buffer"]->set(buffer);


/*
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
    optix::Buffer planBuffer = OGeo::CreateInputUserBuffer<float>( context, planBuf,  4*4, "planBuffer", verbosity); 
    context["planBuffer"]->setBuffer(planBuffer);
*/


    context->validate();
    context->compile();
    context->launch(0, width, height);


/*
    NPY<float>* npy = NPY<float>::make(width, height, 4) ;
    npy->zero();

    void* ptr = buffer->map() ; 
    npy->read( ptr );
    buffer->unmap(); 

    const char* path = "$TMP/oxrap/intersect_analytic_test.npy";
    std::cerr << "save result npy to " << path << std::endl ; 
 
    npy->save(path);
*/


    return 0;
}
