#include "OptiXTest.hh"

#include "SSys.hh"
#include "S_freopen_redirect.hh"

#include "SDirect.hh"
#include "NPY.hpp"

#include "OPTICKS_LOG.hh"


int main( int argc, char** argv ) 
{
    OPTICKS_LOG(argc, argv);

    optix::Context context = optix::Context::create();

    const char* buildrel = "optixrap/tests" ; 
    const char* cmake_target = "redirectLogTest" ; 
    OptiXTest* test = new OptiXTest(context, "redirectLogTest.cu", "minimal", "exception", buildrel, cmake_target ) ;
    test->Summary(argv[0]);

    //unsigned width = 512 ; 
    //unsigned height = 512 ; 

    unsigned width = 16 ; 
    unsigned height = 16 ; 



    // optix::Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width, height );
    optix::Buffer buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, width*height );
    context["output_buffer"]->set(buffer);

    context->validate();
    context->compile();


/*
    // hmm this fails to capture anything
    // lower level freopen redirect works 

    std::stringstream coutbuf ;
    std::stringstream cerrbuf ;
    {    
         cout_redirect out_(coutbuf.rdbuf());
         cerr_redirect err_(cerrbuf.rdbuf()); 

         context->launch(0, width, height);
    }    
    std::string out = coutbuf.str();
    std::string err = cerrbuf.str();


    LOG(info) << " captured out " << out.size() << " err " << err.size() ; 
    LOG(info) << "out("<< out.size() << "):\n" << out ; 
    LOG(info) << "err("<< err.size() << "):\n" << err ; 
*/


    const char* path = "/tmp/redirectLogTest.log" ; 

    SSys::Dump(path);
    {
        S_freopen_redirect sfr(stdout, path );
        context->launch(0, width, height);
        SSys::Dump(path);
    }
    SSys::Dump(path);



    NPY<float>* npy = NPY<float>::make(width, height, 4) ;
    npy->zero();

    void* ptr = buffer->map() ; 
    npy->read( ptr );
    buffer->unmap(); 

    const char* bufpath = "$TMP/redirectLogTest.npy";
    std::cerr << "save result npy to " << bufpath << std::endl ; 
 
    npy->save(bufpath);


    return 0;
}
